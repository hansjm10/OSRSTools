// src/server/index.ts
import express from 'express';
import cors from 'cors';
import OSRSMarketTrainingSystem from "../training/OSRSMarketTrainingSystem";
import {FlipPrediction} from "../types";

// Set default environment variables
process.env.ANALYZE_ALL_ITEMS = process.env.ANALYZE_ALL_ITEMS || 'true';
process.env.INITIAL_ITEM_COUNT = process.env.INITIAL_ITEM_COUNT || '50';
process.env.BATCH_SIZE = process.env.BATCH_SIZE || '10';
process.env.UPDATE_INTERVAL = process.env.UPDATE_INTERVAL || '30'; // minutes
process.env.TOP_PREDICTION_COUNT = process.env.TOP_PREDICTION_COUNT || '10';

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Initialize our market system
const marketSystem = new OSRSMarketTrainingSystem();

// Check if GPU acceleration is available
(async () => {
    try {
        console.log('Checking GPU/Metal acceleration status...');
        await marketSystem.predictor.checkGPUUsage();
    } catch (error) {
        console.error('Error checking GPU status:', error);
    }
})();

// Keep track of predictions and analysis results
let lastPredictions: {
    bestBuys: FlipPrediction[];
    bestSells: FlipPrediction[];
    timestamp: number;
} = {
    bestBuys: [],
    bestSells: [],
    timestamp: 0
};

// Keep track of training results
let modelAccuracies: {
    [itemId: string]: {
        itemName: string;
        accuracy: number;
        mae: number;
        rmse: number;
    };
} = {};

// Track historical prices for key items
const priceHistory: {
    date: string;
    prices: {
        [itemId: string]: number;
    };
}[] = [];

// Track system performance metrics
let systemPerformance = {
    dailyPredictions: 0,
    accuracyRate: 0,
    avgConfidence: 0,
    totalItemsTracked: 0,
    trainingTime: 0,
    lastUpdate: new Date().toISOString()
};

// Initialize system
async function initializeSystem() {
    try {
        console.log('Initializing OSRS Market System...');

        // Get configuration from environment variables
        const analyzeAllItems = process.env.ANALYZE_ALL_ITEMS === 'true';
        const initialItemCount = parseInt(process.env.INITIAL_ITEM_COUNT || '50', 10);
        const batchSize = parseInt(process.env.BATCH_SIZE || '10', 10);
        const updateInterval = parseInt(process.env.UPDATE_INTERVAL || '30', 10); // minutes
        
        console.log(`Configuration: analyze all items: ${analyzeAllItems}, initial items: ${initialItemCount}`);

        // Start with top items for immediate results
        let itemIds = await marketSystem.initializeWithTopItems(initialItemCount);
        systemPerformance.totalItemsTracked = itemIds.length;

        // Start training the model on high-value items
        console.log('Starting initial training...');
        const startTime = Date.now();

        // Use batch processing for more efficient API usage
        await marketSystem.processBatch(itemIds, 'train', batchSize);
        
        systemPerformance.trainingTime = Math.round((Date.now() - startTime) / 1000);
        console.log(`Initial training completed in ${systemPerformance.trainingTime}s`);

        // Get accuracy metrics for trained items
        let totalAccuracy = 0;
        let evaluatedItems = 0;

        for (const itemId of itemIds) {
            try {
                const itemName = await marketSystem.dataService.getItemName(itemId);
                const accuracy = await marketSystem.predictor.evaluateAccuracy(itemId);

                modelAccuracies[itemId] = {
                    itemName,
                    accuracy: accuracy.accuracy,
                    mae: accuracy.mae,
                    rmse: accuracy.rmse
                };

                totalAccuracy += accuracy.accuracy;
                evaluatedItems++;
            } catch (error) {
                console.error(`Could not evaluate accuracy for item ${itemId}:`, error);
            }
        }

        // Calculate average accuracy
        systemPerformance.accuracyRate = evaluatedItems > 0 ? totalAccuracy / evaluatedItems : 0;

        // Generate initial predictions
        await updatePredictions();

        // If we want to analyze all items, schedule background training for remaining items
        if (analyzeAllItems) {
            console.log('Scheduling background training for additional items...');
            
            // Run in the background to avoid blocking server startup
            setTimeout(async () => {
                try {
                    // Get all tradeable items
                    const allItems = await marketSystem.getAllTradeableItems();
                    
                    // Filter out items we've already trained
                    const remainingItems = allItems.filter(id => !itemIds.includes(id));
                    console.log(`Training ${remainingItems.length} additional items in the background...`);
                    
                    // Process in smaller batches to avoid overwhelming the API
                    const smallBatchSize = Math.min(5, batchSize);
                    
                    // This will run in the background and not block
                    marketSystem.processBatch(remainingItems, 'train', smallBatchSize)
                        .then(trainedItems => {
                            console.log(`Background training completed for ${trainedItems.length} additional items`);
                            systemPerformance.totalItemsTracked += trainedItems.length;
                        })
                        .catch(err => console.error('Error in background training:', err));
                } catch (error) {
                    console.error('Error scheduling background training:', error);
                }
            }, 60000); // Start after 1 minute to let the server stabilize
        }

        // Schedule regular updates
        setInterval(updatePredictions, updateInterval * 60 * 1000);

        console.log('System initialization complete');
    } catch (error) {
        console.error('Failed to initialize system:', error);
    }
}

// Function to update predictions
async function updatePredictions() {
    try {
        const updateStartTime = Date.now();
        console.log('⏳ Starting market prediction update...');
        
        // Get prediction count from env or default to 10
        const topPredictionCount = parseInt(process.env.TOP_PREDICTION_COUNT || '10', 10);

        // Find best flipping opportunities (uses our new system that handles all items)
        const recommendations = await marketSystem.findBestFlips(topPredictionCount);

        // Update our stored predictions
        lastPredictions = {
            bestBuys: recommendations.bestBuys,
            bestSells: recommendations.bestSells,
            timestamp: Date.now()
        };

        // Calculate average confidence
        const allPredictions = [...recommendations.bestBuys, ...recommendations.bestSells];
        const totalConfidence = allPredictions.reduce((sum, pred) => sum + (pred.confidence || 0), 0);
        systemPerformance.avgConfidence = allPredictions.length > 0 ? totalConfidence / allPredictions.length : 0;

        // Update prediction count
        systemPerformance.dailyPredictions = allPredictions.length;

        // Update last update time
        systemPerformance.lastUpdate = new Date().toISOString();

        // Update price history for top items (track both buy and sell recommendations)
        const today = new Date().toISOString().split('T')[0];
        const topItemIds = [
            ...recommendations.bestBuys.slice(0, 5).map(item => item.itemId),
            ...recommendations.bestSells.slice(0, 5).map(item => item.itemId)
        ];

        // Get the current prices
        const prices: { [itemId: string]: number } = {};

        // Process in parallel for efficiency (silently)
        await Promise.all(topItemIds.map(async (itemId) => {
            try {
                const metrics = await marketSystem.analyzer.analyzeItem(itemId);
                prices[itemId] = metrics.currentPrice;
            } catch (error) {
                // Silent error handling
            }
        }));

        // Add to price history, limiting to last 30 days
        priceHistory.push({ date: today, prices });
        if (priceHistory.length > 30) {
            priceHistory.shift();
        }

        // Calculate processing time and log completion
        const processingTime = Math.round((Date.now() - updateStartTime) / 1000);
        console.log(`✅ Market update complete in ${processingTime}s - Found ${recommendations.bestBuys.length} buys, ${recommendations.bestSells.length} sells`);
        
        // Log trained items count
        const trainedCount = marketSystem.trainedItems.size;
        const avgConfidencePercent = Math.round(systemPerformance.avgConfidence * 100);
        console.log(`🧠 System status: ${trainedCount} trained item models, ${avgConfidencePercent}% avg confidence`);
    } catch (error) {
        console.error('❌ Error updating predictions:', error);
    }
}

// API Routes

// Get system status and performance metrics
app.get('/api/system/status', async (req, res) => {
    try {
        // Get counts of different item categories
        const allTradeableItems = await marketSystem.getAllTradeableItems();
        const trainedItemsCount = marketSystem.trainedItems?.size || 0;
        
        res.json({
            status: 'online',
            performance: systemPerformance,
            items: {
                total: allTradeableItems.length,
                trained: trainedItemsCount,
                tracked: systemPerformance.totalItemsTracked,
                pendingTraining: allTradeableItems.length - trainedItemsCount,
                analyzeAllEnabled: process.env.ANALYZE_ALL_ITEMS === 'true'
            },
            lastUpdate: systemPerformance.lastUpdate,
            avgConfidence: systemPerformance.avgConfidence
        });
    } catch (error) {
        console.error('Error generating system status:', error);
        res.status(500).json({
            status: 'error',
            error: 'Failed to generate complete system status'
        });
    }
});

// Get current buy/sell recommendations
app.get('/api/recommendations', (req, res) => {
    res.json({
        recommendations: {
            bestBuys: lastPredictions.bestBuys,
            bestSells: lastPredictions.bestSells
        },
        timestamp: lastPredictions.timestamp,
        lastUpdate: new Date(lastPredictions.timestamp).toISOString()
    });
});

// Get detailed model accuracy information
app.get('/api/analytics/accuracy', (req, res) => {
    const accuracyData = Object.entries(modelAccuracies).map(([itemId, data]) => ({
        itemId,
        itemName: data.itemName,
        accuracy: data.accuracy,
        mae: data.mae,
        rmse: data.rmse
    }));

    res.json({
        accuracyData,
        averageAccuracy: systemPerformance.accuracyRate
    });
});

// Get price history data
app.get('/api/analytics/price-history', (req, res) => {
    // Transform the data format to make it easier to use with charts
    const formattedHistory = priceHistory.map(entry => {
        const dataPoint: any = { date: entry.date };

        // Add each item's price as a property
        Object.entries(entry.prices).forEach(([itemId, price]) => {
            const itemName = modelAccuracies[itemId]?.itemName || `Item ${itemId}`;
            dataPoint[itemName] = price;
        });

        return dataPoint;
    });

    res.json({
        priceHistory: formattedHistory,
        itemIds: Object.keys(priceHistory[0]?.prices || {})
    });
});

// Get all trained items with their status
app.get('/api/items', async (req, res) => {
    try {
        // Parse query parameters
        const limit = parseInt(req.query.limit as string || '100', 10);
        const offset = parseInt(req.query.offset as string || '0', 10);
        const sort = (req.query.sort as string) || 'value';
        const sortDir = (req.query.dir as string) || 'desc';
        
        // Get all tradeable items
        const allItems = await marketSystem.getAllTradeableItems();
        const bulkData = await marketSystem.dataService.getBulkData();
        
        // Map items to include additional data
        const itemsWithDetails = await Promise.all(
            allItems.map(async (itemId) => {
                try {
                    const item = bulkData[itemId] as any;
                    
                    if (!item || typeof item !== 'object') {
                        return null;
                    }
                    
                    // Check if this item has been trained
                    const isTrained = marketSystem.trainedItems?.has(itemId) || false;
                    
                    return {
                        itemId,
                        name: item.name,
                        value: item.value || 0,
                        limit: item.limit || 0,
                        members: item.members || false,
                        isTrained,
                        accuracy: isTrained ? (modelAccuracies[itemId]?.accuracy || 0) : 0
                    };
                } catch (error) {
                    console.warn(`Error processing item ${itemId}:`, error);
                    return null;
                }
            })
        );
        
        // Filter out null results
        const validItems = itemsWithDetails.filter(item => item !== null);
        
        // Sort the items
        validItems.sort((a, b) => {
            // @ts-ignore - TypeScript might complain about a[sort] but it's fine
            const valA = a[sort];
            // @ts-ignore
            const valB = b[sort];
            
            // Handle string vs number comparison
            if (typeof valA === 'string' && typeof valB === 'string') {
                return sortDir === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
            } else {
                return sortDir === 'asc' ? valA - valB : valB - valA;
            }
        });
        
        // Apply pagination
        const paginatedItems = validItems.slice(offset, offset + limit);
        
        res.json({
            items: paginatedItems,
            total: validItems.length,
            trained: validItems.filter(item => item.isTrained).length,
            page: {
                offset,
                limit,
                hasMore: offset + limit < validItems.length
            }
        });
    } catch (error) {
        console.error('Error fetching items:', error);
        res.status(500).json({ error: 'Failed to fetch items data' });
    }
});

// Get detailed information for a specific item
app.get('/api/items/:itemId', async (req, res) => {
    const { itemId } = req.params;

    try {
        const itemName = await marketSystem.dataService.getItemName(itemId);
        const metrics = await marketSystem.analyzer.analyzeItem(itemId);
        const isTrained = marketSystem.trainedItems?.has(itemId) || false;

        // Get historical data
        const history = await marketSystem.dataService.getItemHistory(itemId);
        
        // Get additional item info from bulk data
        const bulkData = await marketSystem.dataService.getBulkData();
        const item = bulkData[itemId] as any;

        res.json({
            itemId,
            itemName,
            metrics,
            history: history.slice(-100), // Return last 100 data points
            modelAccuracy: modelAccuracies[itemId] || null,
            isTrained,
            details: {
                value: item?.value || 0,
                limit: item?.limit || 0,
                members: item?.members || false,
                examine: item?.examine || ''
            }
        });
    } catch (error) {
        console.error(`Error fetching item ${itemId}:`, error);
        res.status(500).json({ error: 'Failed to fetch item data' });
    }
});

// Manually trigger a system update
app.post('/api/system/update', async (req, res) => {
    try {
        await updatePredictions();
        res.json({
            success: true,
            message: 'System updated successfully',
            timestamp: lastPredictions.timestamp
        });
    } catch (error) {
        console.error('Error during manual update:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to update system'
        });
    }
});

// Start the backend server
app.listen(PORT, () => {
    console.log(`OSRS Market Backend running on port ${PORT}`);

    // Initialize our system after server starts
    initializeSystem();
});

export default app;