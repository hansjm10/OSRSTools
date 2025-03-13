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

        // Filter items and sort by trading volume
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
                // Sort by item value (higher value items first)
                return itemB.value - itemA.value;
            })
            .slice(0, count); // Take only the top N items

        console.log(`Selected ${itemIds.length} items for training:`);

        // Log the selected items
        for (const itemId of itemIds) {
            const name = await this.dataService.getItemName(itemId);
            console.log(`- ${name} (${itemId})`);
        }

        return itemIds;
    }

    /**
     * Train the model on the specified items
     */
    async train(itemIds: string[], options: {
        trainingPeriods?: number,
        validationSplit?: number,
        predictionHorizon?: number
    } = {}): Promise<void> {
        const config = {
            trainingPeriods: options.trainingPeriods || 30,
            validationSplit: options.validationSplit || 0.2,
            predictionHorizon: options.predictionHorizon || 24
        };

        console.log(`Starting market training with ${itemIds.length} items...`);

        // Train each item
        for (const itemId of itemIds) {
            await this.trainItem(itemId, config);
        }

        console.log('Training complete. Testing model accuracy...');

        // Evaluate accuracy for each trained item
        for (const itemId of itemIds) {
            try {
                const name = await this.dataService.getItemName(itemId);
                const accuracy = await this.predictor.evaluateAccuracy(itemId);
                console.log(`Model accuracy for ${name} (${itemId}):`);
                console.log(`- MAE: ${accuracy.mae.toFixed(4)}`);
                console.log(`- RMSE: ${accuracy.rmse.toFixed(4)}`);
                console.log(`- Directional accuracy: ${(accuracy.accuracy * 100).toFixed(2)}%`);
            } catch (error) {
                console.error(`Could not evaluate accuracy for item ${itemId}:`, error);
            }
        }
    }

    /**
     * Train model on a specific item
     */
    private async trainItem(itemId: string, config: {
        trainingPeriods: number,
        validationSplit: number,
        predictionHorizon: number
    }): Promise<void> {
        const name = await this.dataService.getItemName(itemId);
        console.log(`Training model for ${name} (${itemId})...`);

        try {
            // Get full historical data for this item
            const history = await this.dataService.getItemHistory(itemId);

            // We need enough historical data to train
            if (history.length < config.trainingPeriods + 1) {
                console.warn(`Not enough historical data for ${name} (${itemId}). Skipping.`);
                return;
            }

            // Sort history by timestamp to ensure chronological order
            history.sort((a, b) => a.timestamp - b.timestamp);

            // Determine how many training examples we can create
            const maxExamples = Math.min(
                config.trainingPeriods,
                history.length - Math.ceil(config.predictionHorizon / 24) // Ensure we have future data
            );

            console.log(`Creating ${maxExamples} training examples for ${name}...`);

            // Collect training examples from historical data
            for (let i = 0; i < maxExamples; i++) {
                try {
                    // Calculate the starting point for this training example
                    const startIndex = Math.floor(i * (history.length - maxExamples) / maxExamples);

                    // Create a slice of history data for this training example
                    const trainingEndIndex = startIndex + Math.floor(history.length * 0.7); // Use 70% of data for analysis
                    const trainingSlice = history.slice(startIndex, trainingEndIndex);

                    if (trainingSlice.length === 0) {
                        console.warn(`Empty training slice for example ${i+1}. Skipping.`);
                        continue;
                    }

                    const currentTime = trainingSlice[trainingSlice.length - 1].timestamp;

                    // Analyze market metrics for this slice
                    const metrics = await this.analyzer.analyzeItem(itemId);

                    // Get the actual outcome after the prediction horizon
                    const actualOutcome = await this.dataService.getItemOutcome(
                        itemId,
                        currentTime,
                        config.predictionHorizon
                    );

                    // Add this example to the training data
                    await this.predictor.addTrainingExample(itemId, metrics, actualOutcome);

                    console.log(`Added training example ${i+1}/${maxExamples} for ${name} (${itemId})`);
                } catch (error) {
                    console.error(`Error with training example ${i+1} for ${name}:`, error);
                }
            }

            console.log(`Training for ${name} (${itemId}) complete.`);
        } catch (error) {
            console.error(`Error training ${name} (${itemId}):`, error);
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
                const predictedChange = await this.predictor.predict(metrics);

                // Calculate buy and sell recommendations based on the predicted price change
                const currentPrice = metrics.currentPrice;
                const buyRecommendation = predictedChange > 0 ?
                    Math.min(Math.abs(predictedChange) / currentPrice, 1) : 0;

                const sellRecommendation = predictedChange < 0 ?
                    Math.min(Math.abs(predictedChange) / currentPrice, 1) : 0;

                // Calculate confidence based on model accuracy for this item
                let confidence = 0.5; // Default confidence
                try {
                    const accuracy = await this.predictor.evaluateAccuracy(itemId);
                    confidence = accuracy.accuracy;
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
     * Find the best items to buy and sell based on predictions
     */
    async findBestFlips(count: number = 5): Promise<{
        bestBuys: FlipPrediction[],
        bestSells: FlipPrediction[]
    }> {
        // Get top items to analyze
        const itemIds = await this.initializeWithTopItems(50);

        // Generate predictions for all items
        const predictions = await this.generatePredictions(itemIds);

        // Filter out predictions with low confidence
        const confidencePredictions = predictions.filter(p => p.confidence > 0.55);

        // Sort by buy and sell recommendations
        const bestBuys = confidencePredictions
            .filter(p => p.buyRecommendation > 0.01)
            .sort((a, b) => {
                // Sort by confidence-adjusted buy recommendation
                const scoreA = a.buyRecommendation * a.confidence;
                const scoreB = b.buyRecommendation * b.confidence;
                return scoreB - scoreA;
            })
            .slice(0, count);

        const bestSells = confidencePredictions
            .filter(p => p.sellRecommendation > 0.01)
            .sort((a, b) => {
                // Sort by confidence-adjusted sell recommendation
                const scoreA = a.sellRecommendation * a.confidence;
                const scoreB = b.sellRecommendation * b.confidence;
                return scoreB - scoreA;
            })
            .slice(0, count);

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

        // Train the model
        await trainingSystem.train(itemIds, {
            trainingPeriods: 20,
            predictionHorizon: 24 // 24 hours
        });

        // Find best flipping opportunities
        const { bestBuys, bestSells } = await trainingSystem.findBestFlips(5);

        // Display results
        console.log('\nTop 5 Recommended Buys:');
        bestBuys.forEach((prediction, index) => {
            console.log(`${index + 1}. ${prediction.itemName} (${prediction.itemId}):`);
            console.log(`   Current Price: ${prediction.metrics.currentPrice.toLocaleString()} gp`);
            console.log(`   Predicted Change: +${prediction.predictedPriceChange.toFixed(0)} gp (${(prediction.buyRecommendation * 100).toFixed(2)}%)`);
            console.log(`   Confidence: ${(prediction.confidence * 100).toFixed(2)}%`);
            console.log(`   Projected Profit: ${(prediction.predictedPriceChange).toFixed(0)} gp per item`);
        });

        console.log('\nTop 5 Recommended Sells:');
        bestSells.forEach((prediction, index) => {
            console.log(`${index + 1}. ${prediction.itemName} (${prediction.itemId}):`);
            console.log(`   Current Price: ${prediction.metrics.currentPrice.toLocaleString()} gp`);
            console.log(`   Predicted Change: ${prediction.predictedPriceChange.toFixed(0)} gp (${(prediction.sellRecommendation * 100).toFixed(2)}%)`);
            console.log(`   Confidence: ${(prediction.confidence * 100).toFixed(2)}%`);
            console.log(`   Projected Savings: ${Math.abs(prediction.predictedPriceChange).toFixed(0)} gp per item`);
        });
    } catch (error) {
        console.error('Error running OSRS market training:', error);
    }
}

// Uncomment to run the training
// runOSRSMarketTraining().catch(console.error);

export default OSRSMarketTrainingSystem;