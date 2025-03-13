// src/server/index.ts
import express from 'express';
import cors from 'cors';
import OSRSMarketTrainingSystem from "../training/OSRSMarketTrainingSystem";
import {FlipPrediction} from "../types";

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Initialize our market system
const marketSystem = new OSRSMarketTrainingSystem();

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

        // Get top items for training
        const itemIds = await marketSystem.initializeWithTopItems(20);
        systemPerformance.totalItemsTracked = itemIds.length;

        // Train the model on these items
        console.log('Starting initial training...');
        const startTime = Date.now();

        await marketSystem.train(itemIds, {
            trainingPeriods: 20,
            predictionHorizon: 24
        });

        systemPerformance.trainingTime = Math.round((Date.now() - startTime) / 1000);
        console.log(`Training completed in ${systemPerformance.trainingTime}s`);

        // Get accuracy metrics for each item
        let totalAccuracy = 0;

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
            } catch (error) {
                console.error(`Could not evaluate accuracy for item ${itemId}:`, error);
            }
        }

        // Calculate average accuracy
        systemPerformance.accuracyRate = totalAccuracy / Object.keys(modelAccuracies).length;

        // Generate predictions
        await updatePredictions();

        // Schedule regular updates
        setInterval(updatePredictions, 30 * 60 * 1000); // Update every 30 minutes

        console.log('System initialization complete');
    } catch (error) {
        console.error('Failed to initialize system:', error);
    }
}

// Function to update predictions
async function updatePredictions() {
    try {
        console.log('Updating market predictions...');

        // Find best flipping opportunities
        const recommendations = await marketSystem.findBestFlips(10);

        // Update our stored predictions
        lastPredictions = {
            bestBuys: recommendations.bestBuys,
            bestSells: recommendations.bestSells,
            timestamp: Date.now()
        };

        // Calculate average confidence
        const allPredictions = [...recommendations.bestBuys, ...recommendations.bestSells];
        const totalConfidence = allPredictions.reduce((sum, pred) => sum + (pred.confidence || 0), 0);
        systemPerformance.avgConfidence = totalConfidence / allPredictions.length;

        // Update daily prediction count
        systemPerformance.dailyPredictions = allPredictions.length;

        // Update last update time
        systemPerformance.lastUpdate = new Date().toISOString();

        // Update price history for top items
        const today = new Date().toISOString().split('T')[0];
        const topItemIds = [
            ...recommendations.bestBuys.slice(0, 3).map(item => item.itemId),
            ...recommendations.bestSells.slice(0, 3).map(item => item.itemId)
        ];

        const prices: { [itemId: string]: number } = {};

        for (const itemId of topItemIds) {
            try {
                const metrics = await marketSystem.analyzer.analyzeItem(itemId);
                prices[itemId] = metrics.currentPrice;
            } catch (error) {
                console.error(`Error getting current price for item ${itemId}:`, error);
            }
        }

        // Add to price history, limiting to last 30 days
        priceHistory.push({ date: today, prices });
        if (priceHistory.length > 30) {
            priceHistory.shift();
        }

        console.log('Predictions updated successfully');
    } catch (error) {
        console.error('Error updating predictions:', error);
    }
}

// API Routes

// Get system status and performance metrics
app.get('/api/system/status', (req, res) => {
    res.json({
        status: 'online',
        performance: systemPerformance,
        itemsTracked: systemPerformance.totalItemsTracked
    });
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

// Get detailed information for a specific item
app.get('/api/items/:itemId', async (req, res) => {
    const { itemId } = req.params;

    try {
        const itemName = await marketSystem.dataService.getItemName(itemId);
        const metrics = await marketSystem.analyzer.analyzeItem(itemId);

        // Get historical data
        const history = await marketSystem.dataService.getItemHistory(itemId);

        res.json({
            itemId,
            itemName,
            metrics,
            history: history.slice(-100), // Return last 100 data points
            modelAccuracy: modelAccuracies[itemId] || null
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