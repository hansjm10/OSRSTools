// src/training/OSRSMarketTrainingSystem.ts

import {FlipPrediction, TimeseriesData} from "../types";
import {MarketAnalyzer} from "../analysis/MarketAnalyzer";
import {MarketMLPredictor} from "../ml/MarketMLPredictor";
import {BulkDataResponse, OSRSDataService} from "../services/OSRSDataService";

interface OSRSDataServiceAdapter {
    getItemHistory(itemId: string): Promise<TimeseriesData[]>;
    getItemOutcome(itemId: string, fromTimestamp: number, horizon: number): Promise<number>;
    getBulkData(): Promise<Record<string, any>>;
    getItemName(itemId: string): Promise<string>;
}

interface TrainingOptions {
    trainingPeriods?: number;
    validationSplit?: number;
    predictionHorizon?: number;
    batchSize?: number;
    epochs?: number;
    earlyStoppingPatience?: number;
}

/**
 * OSRS Market Training System - Connects to real OSRS market data
 * and trains market prediction models.
 */
export class OSRSMarketTrainingSystem {
    dataService: OSRSDataServiceAdapter;
    analyzer: MarketAnalyzer;
    predictor: MarketMLPredictor;
    private bulkData: BulkDataResponse | null = null;
    private itemHistoryCache: Map<string, TimeseriesData[]> = new Map();
    trainedItems: Set<string> = new Set(); // Make public so we can access from server

    constructor() {
        const osrsDataService = new OSRSDataService();
        this.dataService = this.createDataServiceAdapter(osrsDataService);
        this.analyzer = new MarketAnalyzer(this.dataService);
        this.predictor = new MarketMLPredictor();
    }

    /**
     * Create a data service adapter that conforms to the interface
     * expected by MarketTrainingSystem but uses OSRSDataService
     */
    private createDataServiceAdapter(osrsService: OSRSDataService): OSRSDataServiceAdapter {
        return {
            getItemHistory: async (itemId: string): Promise<TimeseriesData[]> => {
                // Check cache first
                if (this.itemHistoryCache.has(itemId)) {
                    return this.itemHistoryCache.get(itemId)!;
                }

                // Fetch and cache the data
                const history = await osrsService.getItemHistory(itemId);
                this.itemHistoryCache.set(itemId, history);
                return history;
            },

            getItemOutcome: async (itemId: string, fromTimestamp: number, horizon: number): Promise<number> => {
                // Ensure we have the historical data
                const history = await this.dataService.getItemHistory(itemId);

                // Find the data point closest to fromTimestamp
                const currentIndex = history.findIndex(d => d.timestamp >= fromTimestamp);
                if (currentIndex === -1) {
                    throw new Error(`No data point found at or after timestamp ${fromTimestamp}`);
                }

                // Calculate the target timestamp (horizon hours later)
                const futureTimestamp = fromTimestamp + (horizon * 3600);

                // Find the closest future data point
                let futureIndex = history.findIndex(d => d.timestamp >= futureTimestamp);

                // If we can't find a future point, use the last available point
                if (futureIndex === -1) {
                    futureIndex = history.length - 1;
                    console.warn(`Warning: Using last available data point for item ${itemId} instead of horizon ${horizon} hours`);
                }

                // Return the price difference
                return history[futureIndex].price - history[currentIndex].price;
            },

            getBulkData: async (): Promise<Record<string, any>> => {
                if (!this.bulkData) {
                    this.bulkData = await osrsService.getBulkData();
                }

                // Convert to the format expected by MarketAnalyzer
                const formattedData: Record<string, any> = {};

                for (const [key, value] of Object.entries(this.bulkData)) {
                    // Skip timestamp entries (they're numbers)
                    if (typeof value === 'number') continue;

                    formattedData[key] = {
                        id: key,
                        name: value.name,
                        limit: value.limit || 0,
                        members: value.members,
                        latestPrice: value.value
                    };
                }

                return formattedData;
            },

            getItemName: async (itemId: string): Promise<string> => {
                if (!this.bulkData) {
                    this.bulkData = await osrsService.getBulkData();
                }

                const item = this.bulkData[itemId];
                if (typeof item === 'object' && item !== null) {
                    return item.name || `Item ${itemId}`;
                }

                return `Item ${itemId}`;
            }
        };
    }

    /**
     * Initialize training with the most tradable items in OSRS
     */
    async initializeWithTopItems(count: number = 20): Promise<string[]> {
        console.log(`Initializing with top ${count} tradable items...`);

        // Get all items from bulk data
        const bulkData = await this.dataService.getBulkData();

        // Filter items and sort by trading volume and value
        const itemIds = Object.keys(bulkData)
            .filter(key => {
                const item = bulkData[key];
                // Make sure it's an item object, not a timestamp
                return typeof item === 'object' && item !== null &&
                    // Only include members items with reasonable limits
                    item.members === true && item.limit > 0;
            })
            .sort((a, b) => {
                const itemA = bulkData[a] as any;
                const itemB = bulkData[b] as any;
                // Sort items by a combination of value and limit (tradability)
                const scoreA = itemA.value * Math.min(100, itemA.limit || 0);
                const scoreB = itemB.value * Math.min(100, itemB.limit || 0);
                return scoreB - scoreA;
            })
            .slice(0, count); // Take only the top N items

        console.log(`Selected ${itemIds.length} items for analysis:`);

        // Log the selected items
        for (const itemId of itemIds) {
            const name = await this.dataService.getItemName(itemId);
            console.log(`- ${name} (${itemId})`);
        }

        return itemIds;
    }

    /**
     * Get data for cross-validation training approach
     */
    private async prepareTrainingData(
        itemId: string, 
        config: Required<TrainingOptions>
    ): Promise<{
        trainingWindows: {start: number, end: number, outcome: number}[],
        totalPoints: number
    }> {
        // Get full historical data for this item
        const history = await this.dataService.getItemHistory(itemId);
        
        // Sort history by timestamp to ensure chronological order
        history.sort((a, b) => a.timestamp - b.timestamp);
        
        // We need enough historical data to train
        if (history.length < config.trainingPeriods + 5) {
            throw new Error(`Not enough historical data for item ${itemId}. Need at least ${config.trainingPeriods + 5} points, but got ${history.length}`);
        }
        
        // Create sliding windows for training
        const trainingWindows: {start: number, end: number, outcome: number}[] = [];
        
        // Step size for sliding window (move 1/4 of window size each time)
        const windowSize = Math.floor(history.length * 0.7);
        const stepSize = Math.max(1, Math.floor(windowSize / 4));
        
        // Create overlapping windows to maximize training data
        for (let i = 0; i < history.length - windowSize - 1; i += stepSize) {
            const windowStart = i;
            const windowEnd = i + windowSize;
            const predictionPoint = history[windowEnd];
            
            // Calculate future point based on prediction horizon
            const futureTime = predictionPoint.timestamp + (config.predictionHorizon * 3600);
            const futureIndex = history.findIndex(d => d.timestamp >= futureTime);
            
            // Skip if we don't have future data
            if (futureIndex === -1) continue;
            
            const outcome = history[futureIndex].price - predictionPoint.price;
            
            trainingWindows.push({
                start: windowStart,
                end: windowEnd,
                outcome: outcome
            });
        }
        
        return {
            trainingWindows,
            totalPoints: history.length
        };
    }

    /**
     * Train the model on the specified items
     */
    async train(itemIds: string[], options: TrainingOptions = {}): Promise<void> {
        const config: Required<TrainingOptions> = {
            trainingPeriods: options.trainingPeriods || 50,
            validationSplit: options.validationSplit || 0.2,
            predictionHorizon: options.predictionHorizon || 24,
            batchSize: options.batchSize || 16,
            epochs: options.epochs || 50,
            earlyStoppingPatience: options.earlyStoppingPatience || 5
        };

        // Track successes
        const successfulItems: string[] = [];
        const startTime = Date.now();

        // Only log when number of items is worth tracking (batch size)
        if (itemIds.length > 5) {
            console.log(`Training batch of ${itemIds.length} items...`);
        }

        // Train each item
        for (const itemId of itemIds) {
            try {
                await this.trainItem(itemId, config);
                successfulItems.push(itemId);
                this.trainedItems.add(itemId);
            } catch (error) {
                // Silent error handling
            }
        }

        // For each successfully trained item, evaluate accuracy
        for (const itemId of successfulItems) {
            try {
                await this.predictor.evaluateAccuracy(itemId);
            } catch (error) {
                // Silent error handling
            }
        }

        // Only log completion for larger batches
        if (itemIds.length > 5) {
            const duration = ((Date.now() - startTime) / 1000).toFixed(1);
            console.log(`Trained ${successfulItems.length}/${itemIds.length} items in ${duration}s`);
        }
    }

    /**
     * Train model on a specific item
     */
    private async trainItem(itemId: string, config: Required<TrainingOptions>): Promise<void> {
        const name = await this.dataService.getItemName(itemId);

        try {
            // Prepare training data with cross-validation approach
            const { trainingWindows, totalPoints } = await this.prepareTrainingData(itemId, config);
            
            if (trainingWindows.length < 20) {
                // Silent skip for insufficient data
                return;
            }
            
            // Keep this log for tracking progress
            console.log(`Training ${name} (${itemId}): ${trainingWindows.length} windows`);
            
            // Process each window to create training examples
            for (const window of trainingWindows) {
                try {
                    // Get the timestamp at the end of the window
                    const history = await this.dataService.getItemHistory(itemId);
                    const timestamp = history[window.end].timestamp;
                    
                    // Analyze market metrics at this point
                    const metrics = await this.analyzer.analyzeItem(itemId);
                    
                    // Add this example with the known outcome
                    await this.predictor.addTrainingExample(itemId, metrics, window.outcome);
                } catch (error) {
                    // Silent error handling for individual windows
                }
            }
            
            // Force a retrain now that we have all examples
            await this.predictor.retrainAll();
        } catch (error: unknown) {
            // Log only critical errors
            const errorMessage = error instanceof Error ? error.message : String(error);
            console.error(`Error training ${itemId}:`, errorMessage);
            throw error; // Propagate the error to handle it in the train method
        }
    }

    /**
     * Generate predictions for the specified items
     */
    async generatePredictions(itemIds: string[]): Promise<FlipPrediction[]> {
        const predictions: FlipPrediction[] = [];

        for (const itemId of itemIds) {
            try {
                const name = await this.dataService.getItemName(itemId);
                console.log(`Generating prediction for ${name} (${itemId})...`);

                const metrics = await this.analyzer.analyzeItem(itemId);
                
                // Use item-specific model if available
                const predictedChange = this.trainedItems.has(itemId) ?
                    await this.predictor.predict(metrics, itemId) :
                    await this.predictor.predict(metrics);

                // Calculate buy and sell recommendations based on the predicted price change
                const currentPrice = metrics.currentPrice;
                const predictedPercentage = predictedChange / currentPrice;
                
                const buyRecommendation = predictedPercentage > 0 ?
                    Math.min(Math.abs(predictedPercentage), 1) : 0;

                const sellRecommendation = predictedPercentage < 0 ?
                    Math.min(Math.abs(predictedPercentage), 1) : 0;

                // Calculate confidence based on model accuracy for this item
                let confidence = 0.5; // Default confidence
                try {
                    if (this.trainedItems.has(itemId)) {
                        const accuracy = await this.predictor.evaluateAccuracy(itemId);
                        confidence = accuracy.accuracy;
                    }
                } catch (error) {
                    console.warn(`Could not determine confidence for ${name} (${itemId})`);
                }

                predictions.push({
                    itemId,
                    itemName: name,
                    metrics,
                    buyRecommendation,
                    sellRecommendation,
                    predictedPriceChange: predictedChange,
                    confidence
                });
            } catch (error) {
                console.error(`Error generating prediction for item ${itemId}:`, error);
            }
        }

        return predictions;
    }

    /**
     * Calculate optimal quantity to flip based on item data
     */
    private calculateOptimalQuantity(
        itemId: string, 
        prediction: FlipPrediction, 
        maxBudget: number = 10000000 // Default 10M
    ): number {
        // Get item buy limit (or default to a reasonable value)
        const bulkItem = this.bulkData?.[itemId] as any;
        const buyLimit = (bulkItem?.limit || 100) as number;
        
        // Get price and liquidity metrics
        const price = prediction.metrics.currentPrice;
        const liquidity = prediction.metrics.liquidityScore || 0.5;
        
        // Calculate how many we can afford
        const affordableAmount = Math.floor(maxBudget / price);
        
        // Adjust by liquidity (higher liquidity = can buy closer to limit)
        const liquidityAdjustment = 0.3 + (liquidity * 0.7); // Between 30% and 100% of limit
        const recommendedAmount = Math.floor(buyLimit * liquidityAdjustment);
        
        // Return the lower of what we can afford and what's recommended
        return Math.min(affordableAmount, recommendedAmount);
    }

    /**
     * Get all valid tradeable items from bulk data
     */
    async getAllTradeableItems(): Promise<string[]> {
        console.log('Fetching all tradeable items...');
        
        // Get all items from bulk data
        const bulkData = await this.dataService.getBulkData();
        
        // Filter items to only include tradeable ones
        const itemIds = Object.keys(bulkData)
            .filter(key => {
                const item = bulkData[key];
                // Make sure it's an item object, not a timestamp
                return typeof item === 'object' && item !== null &&
                    // Only include members items with reasonable limits
                    'members' in item && 'limit' in item &&
                    item.members === true && item.limit > 0;
            });
            
        console.log(`Found ${itemIds.length} tradeable items`);
        return itemIds;
    }
    
    /**
     * Batch process items for efficient training and prediction
     */
    async processBatch(itemIds: string[], action: 'train' | 'predict', batchSize: number = 10): Promise<any[]> {
        const results: any[] = [];
        const totalItems = itemIds.length;
        const totalBatches = Math.ceil(totalItems / batchSize);
        
        // Only log at start of processing
        console.log(`${action === 'train' ? 'Training' : 'Predicting'} ${totalItems} items in ${totalBatches} batches`);
        
        // Process in batches to avoid overwhelming the API
        for (let i = 0; i < totalItems; i += batchSize) {
            const batchItemIds = itemIds.slice(i, i + batchSize);
            const batchNum = Math.floor(i / batchSize) + 1;
            
            // Log only every 5 batches to show progress without spamming
            if (batchNum % 5 === 0 || batchNum === 1 || batchNum === totalBatches) {
                console.log(`Batch ${batchNum}/${totalBatches} (${Math.round((batchNum/totalBatches)*100)}%)`);
            }
            
            try {
                if (action === 'train') {
                    await this.train(batchItemIds);
                    results.push(...batchItemIds);
                } else if (action === 'predict') {
                    const predictions = await this.generatePredictions(batchItemIds);
                    results.push(...predictions);
                }
            } catch (error) {
                // Silent error handling
            }
            
            // Add delay between batches to avoid API rate limits
            if (i + batchSize < totalItems) {
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
        
        // Log completion
        console.log(`${action === 'train' ? 'Training' : 'Prediction'} completed: ${results.length}/${totalItems} items processed`);
        
        return results;
    }

    /**
     * Find the best items to buy and sell based on predictions
     */
    async findBestFlips(count: number = 5, maxBudget: number = 10000000): Promise<{
        bestBuys: Array<FlipPrediction & {quantity: number}>,
        bestSells: Array<FlipPrediction & {quantity: number}>
    }> {
        console.log("Finding best flipping opportunities...");
        const startTime = Date.now();
        
        // Get all tradeable items (or use a specified limit)
        const allItems = await this.getAllTradeableItems();
        
        // For performance, we'll prioritize high-value items
        // Sort by value and limit for initial fast results
        console.log(`Sorting ${allItems.length} items by value...`);
        const bulkData = await this.dataService.getBulkData();
        const sortedItems = [...allItems].sort((a, b) => {
            const itemA = bulkData[a] as any;
            const itemB = bulkData[b] as any;
            // Sort items by value and trading limit
            const scoreA = (itemA.value || 0) * Math.min(100, itemA.limit || 0);
            const scoreB = (itemB.value || 0) * Math.min(100, itemB.limit || 0);
            return scoreB - scoreA;
        });
        
        // Get top 100 valuable items first (immediate results)
        const topValueItems = sortedItems.slice(0, 100);
        console.log(`Analyzing top ${topValueItems.length} valuable items...`);
        const predictions = await this.processBatch(topValueItems, 'predict', 10);
        
        // Process remaining items in background (if appropriate setting is enabled)
        if (process.env.ANALYZE_ALL_ITEMS === 'true') {
            const remainingItems = sortedItems.slice(100);
            console.log(`Scheduling background analysis for ${remainingItems.length} additional items...`);
            
            // Process in batches without blocking
            this.processBatch(remainingItems, 'predict', 5)
                .catch(() => {}); // Silent error handling
        }

        // Filter out predictions with low confidence
        const confidencePredictions = predictions.filter(p => p.confidence > 0.55);
        console.log(`Found ${confidencePredictions.length} items with good confidence`);

        // Process buy recommendations
        const bestBuys = confidencePredictions
            .filter(p => p.buyRecommendation > 0.01)
            .sort((a, b) => {
                // Sort by confidence-adjusted buy recommendation
                const scoreA = a.buyRecommendation * a.confidence * Math.min(5000000, a.metrics.averageVolume);
                const scoreB = b.buyRecommendation * b.confidence * Math.min(5000000, b.metrics.averageVolume);
                return scoreB - scoreA;
            })
            .slice(0, count)
            .map(p => ({
                ...p,
                quantity: this.calculateOptimalQuantity(p.itemId, p, maxBudget)
            }));

        // Process sell recommendations
        const bestSells = confidencePredictions
            .filter(p => p.sellRecommendation > 0.01)
            .sort((a, b) => {
                // Sort by confidence-adjusted sell recommendation
                const scoreA = a.sellRecommendation * a.confidence * Math.min(5000000, a.metrics.averageVolume);
                const scoreB = b.sellRecommendation * b.confidence * Math.min(5000000, b.metrics.averageVolume);
                return scoreB - scoreA;
            })
            .slice(0, count)
            .map(p => ({
                ...p,
                quantity: this.calculateOptimalQuantity(p.itemId, p, maxBudget)
            }));

        const duration = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log(`Found ${bestBuys.length} buy and ${bestSells.length} sell recommendations in ${duration}s`);
        
        return { bestBuys, bestSells };
    }
}

// Example usage
async function runOSRSMarketTraining(): Promise<void> {
    try {
        console.log('Initializing OSRS Market Training System...');
        const trainingSystem = new OSRSMarketTrainingSystem();

        // Get top 20 items to train on
        const itemIds = await trainingSystem.initializeWithTopItems(20);

        // Train the model with enhanced options
        await trainingSystem.train(itemIds, {
            trainingPeriods: 50,
            predictionHorizon: 24, // 24 hours
            epochs: 50,
            batchSize: 16,
            earlyStoppingPatience: 5
        });

        // Find best flipping opportunities with a 10M budget
        const { bestBuys, bestSells } = await trainingSystem.findBestFlips(5, 10000000);

        // Display results with quantity recommendations
        console.log('\nTop 5 Recommended Buys:');
        bestBuys.forEach((prediction, index) => {
            console.log(`${index + 1}. ${prediction.itemName} (${prediction.itemId}):`);
            console.log(`   Current Price: ${prediction.metrics.currentPrice.toLocaleString()} gp`);
            console.log(`   Predicted Change: +${prediction.predictedPriceChange.toFixed(0)} gp (${(prediction.buyRecommendation * 100).toFixed(2)}%)`);
            console.log(`   Confidence: ${(prediction.confidence * 100).toFixed(2)}%`);
            console.log(`   Projected Profit: ${(prediction.predictedPriceChange * prediction.quantity).toFixed(0)} gp (${prediction.quantity.toLocaleString()} × ${prediction.predictedPriceChange.toFixed(0)} gp)`);
            console.log(`   Recommended Quantity: ${prediction.quantity.toLocaleString()}`);
            console.log(`   Required Investment: ${(prediction.quantity * prediction.metrics.currentPrice).toLocaleString()} gp`);
        });

        console.log('\nTop 5 Recommended Sells:');
        bestSells.forEach((prediction, index) => {
            console.log(`${index + 1}. ${prediction.itemName} (${prediction.itemId}):`);
            console.log(`   Current Price: ${prediction.metrics.currentPrice.toLocaleString()} gp`);
            console.log(`   Predicted Change: ${prediction.predictedPriceChange.toFixed(0)} gp (${(prediction.sellRecommendation * 100).toFixed(2)}%)`);
            console.log(`   Confidence: ${(prediction.confidence * 100).toFixed(2)}%`);
            console.log(`   Projected Savings: ${Math.abs(prediction.predictedPriceChange * prediction.quantity).toFixed(0)} gp (${prediction.quantity.toLocaleString()} × ${Math.abs(prediction.predictedPriceChange).toFixed(0)} gp)`);
            console.log(`   Recommended Quantity: ${prediction.quantity.toLocaleString()}`);
        });
    } catch (error) {
        console.error('Error running OSRS market training:', error);
    }
}

// Uncomment to run the training
// runOSRSMarketTraining().catch(console.error);

export default OSRSMarketTrainingSystem;