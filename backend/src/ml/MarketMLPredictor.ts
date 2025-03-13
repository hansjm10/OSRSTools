import * as tf from '@tensorflow/tfjs-node';
import {MarketMetrics} from "../types";

interface TrainingData {
    rawInput: number[];
    actualOutcome: number;
    timestamp: number;
}

interface FeatureData {
    tensor: tf.Tensor2D;
    raw: number[];
}

interface ModelAccuracy {
    mae: number;
    rmse: number;
    accuracy: number;
}

export class MarketMLPredictor {
    private model!: tf.LayersModel;
    private readonly tf: typeof tf;
    private trainingData: Map<string, TrainingData[]>;

    constructor() {
        this.tf = tf;
        this.trainingData = new Map();
        this.initModel();
    }

    private async initModel(): Promise<void> {
        // Clear any existing TensorFlow memory/state
        this.tf.disposeVariables();

        // Input layer - explicitly define the dtype
        const input = this.tf.input({
            shape: [15],
            dtype: 'float32',
            name: 'input_layer'
        });

        // First dense layer with explicit weights initialization
        const dense1 = this.tf.layers.dense({
            units: 32,
            activation: 'relu',
            kernelInitializer: 'glorotUniform',
            biasInitializer: 'zeros',
            dtype: 'float32',
            name: 'dense_1'
        }).apply(input) as tf.SymbolicTensor;

        // Dropout layer
        const dropout = this.tf.layers.dropout({
            rate: 0.2,
            name: 'dropout_1'
        }).apply(dense1) as tf.SymbolicTensor;

        // Second dense layer
        const dense2 = this.tf.layers.dense({
            units: 16,
            activation: 'relu',
            kernelInitializer: 'glorotUniform',
            biasInitializer: 'zeros',
            dtype: 'float32',
            name: 'dense_2'
        }).apply(dropout) as tf.SymbolicTensor;

        // Output layer
        const output = this.tf.layers.dense({
            units: 1,
            activation: 'linear',
            kernelInitializer: 'glorotUniform',
            biasInitializer: 'zeros',
            dtype: 'float32',
            name: 'output'
        }).apply(dense2) as tf.SymbolicTensor;

        // Create the model
        this.model = this.tf.model({
            inputs: input,
            outputs: output,
            name: 'market_predictor'
        });

        // Compile the model
        this.model.compile({
            optimizer: this.tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        // Initialize with dummy data - explicitly use float32
        const dummyData = new Array(15).fill(0);
        const dummyInput = this.tf.tensor2d([dummyData], [1, 15], 'float32');

        try {
            // Perform a warm-up prediction and explicitly get the data
            const prediction = this.model.predict(dummyInput) as tf.Tensor;
            await prediction.data();  // Ensure prediction works
            prediction.dispose();
        } catch (error) {
            console.error('Error during model initialization:', error);
            throw new Error(`Failed to initialize model: ${error instanceof Error ? error.message : String(error)}`);
        } finally {
            dummyInput.dispose();
        }
    }

    private metricsToFeatures(metrics: MarketMetrics): FeatureData {
        // Extract only the numeric metrics we need for the model
        // This avoids issues with complex objects like hourlyPatterns and weekdayPatterns

        // Define all required metrics to ensure they exist
        const requiredMetrics = [
            'currentPrice', 'priceVolatility', 'priceVelocity', 'trendStrength',
            'averageVolume', 'volumeVelocity', 'volumeConsistency', 'bidAskSpread',
            'turnoverRate', 'marketImpact', 'sma5', 'sma20', 'macdLine',
            'macdHistogram', 'liquidityScore'
        ];

        // Create a features array from the specific metrics we need
        // This avoids type issues with the complex objects in MarketMetrics
        const features = [
            this.normalizeValue(metrics.currentPrice),
            this.normalizeValue(metrics.priceVolatility),
            this.normalizeValue(metrics.priceVelocity),
            this.normalizeValue(metrics.trendStrength),
            this.normalizeValue(metrics.averageVolume),
            this.normalizeValue(metrics.volumeVelocity),
            this.normalizeValue(metrics.volumeConsistency),
            this.normalizeValue(metrics.bidAskSpread),
            this.normalizeValue(metrics.turnoverRate),
            this.normalizeValue(metrics.marketImpact),
            this.normalizeValue(metrics.sma5),
            this.normalizeValue(metrics.sma20),
            this.normalizeValue(metrics.macdLine),
            this.normalizeValue(metrics.macdHistogram),
            this.normalizeValue(metrics.liquidityScore)
        ];

        // Log the normalized features for debugging
        console.log('Normalized features:', features);

        // Verify all features are valid numbers
        if (!this.validateFeatures(features)) {
            console.error('Invalid features:', features);
            throw new Error('Invalid feature values detected');
        }

        // Always explicitly specify dtype as float32 and shape
        return {
            tensor: this.tf.tensor2d([features], [1, 15], 'float32'),
            raw: features
        };
    }

    private normalizeValue(value: any): number {
        // Handle non-numeric types
        if (typeof value === 'object') {
            console.warn('Received object instead of number, using 0');
            return 0;
        }

        // Check for undefined, null, or NaN
        if (value === undefined || value === null || isNaN(value)) {
            console.warn('Normalizing undefined/null/NaN value to 0');
            return 0;
        }

        // Make sure it's a number type
        const numValue = Number(value);

        // If conversion failed and resulted in NaN, return 0
        if (isNaN(numValue)) {
            console.warn('Failed to convert value to number, using 0');
            return 0;
        }

        // Handle large values
        if (Math.abs(numValue) > 1e6) {
            return numValue / 1e6;
        }

        return numValue;
    }

    private validateFeatures(features: number[]): boolean {
        if (features.length !== 15) {
            console.error(`Feature length mismatch: expected 15, got ${features.length}`);
            return false;
        }

        for (let i = 0; i < features.length; i++) {
            const f = features[i];
            if (typeof f !== 'number' || isNaN(f)) {
                console.error(`Invalid feature at index ${i}: ${f}`);
                return false;
            }
        }

        return true;
    }

    private handleTensorError(error: unknown): never {
        console.error('TensorFlow error:', error);
        throw new Error(`ML prediction failed: ${error instanceof Error ? error.message : String(error)}`);
    }

    private calculateMAE(pred: number[], actual: number[]): number {
        return pred.reduce((sum, p, i) => sum + Math.abs(p - actual[i]), 0) / pred.length;
    }

    private calculateRMSE(pred: number[], actual: number[]): number {
        const mse = pred.reduce((sum, p, i) => sum + Math.pow(p - actual[i], 2), 0) / pred.length;
        return Math.sqrt(mse);
    }

    private calculateDirectionalAccuracy(pred: number[], actual: number[]): number {
        let correct = 0;
        for (let i = 1; i < pred.length; i++) {
            const predDirection = pred[i] > pred[i-1];
            const actualDirection = actual[i] > actual[i-1];
            if (predDirection === actualDirection) correct++;
        }
        return pred.length > 1 ? correct / (pred.length - 1) : 0;
    }

    async predict(metrics: MarketMetrics): Promise<number> {
        // Add debugging to see what's coming in
        console.log('Input metrics:', JSON.stringify(metrics, null, 2));

        const features = this.metricsToFeatures(metrics);
        console.log('Extracted features:', features.raw);

        if (!this.validateFeatures(features.raw)) {
            throw new Error('Invalid feature data');
        }

        // Create a new input tensor with explicitly set dtype
        const inputTensor = this.tf.tensor2d(features.raw, [1, 15], 'float32');

        try {
            // Make sure we're using the right dtype
            const prediction = this.model.predict(inputTensor) as tf.Tensor;
            const value = (await prediction.data())[0];

            // Clean up
            prediction.dispose();

            return value * (Math.abs(metrics.currentPrice) > 1e6 ? 1e6 : 1);
        } catch (error) {
            this.handleTensorError(error);
        } finally {
            // Dispose all tensors to prevent memory leaks
            inputTensor.dispose();
            features.tensor.dispose();
        }
    }

    async addTrainingExample(
        itemId: string,
        metrics: MarketMetrics,
        actualOutcome: number
    ): Promise<void> {
        const features = this.metricsToFeatures(metrics);
        const example: TrainingData = {
            rawInput: features.raw,
            actualOutcome,
            timestamp: Date.now()
        };

        if (!this.trainingData.has(itemId)) {
            this.trainingData.set(itemId, []);
        }
        const itemData = this.trainingData.get(itemId);
        if (itemData) {
            itemData.push(example);
        }

        features.tensor.dispose();

        if (this.trainingData.get(itemId)?.length === 10) {
            await this.retrain(itemId);
        }
    }

    private async retrain(itemId: string): Promise<void> {
        const data = this.trainingData.get(itemId);
        if (!data || data.length === 0) {
            console.warn('No training data available for item:', itemId);
            return;
        }

        // Always explicitly specify dtype
        const inputs = this.tf.tensor2d(data.map(d => d.rawInput), undefined, 'float32');
        const outputs = this.tf.tensor2d(data.map(d => [d.actualOutcome]), undefined, 'float32');

        try {
            await this.model.fit(inputs, outputs, {
                epochs: 10,
                validationSplit: 0.2,
                shuffle: true
            });
        } finally {
            inputs.dispose();
            outputs.dispose();
        }
    }

    async evaluateAccuracy(itemId: string): Promise<ModelAccuracy> {
        const data = this.trainingData.get(itemId) || [];
        if (data.length < 10) {
            throw new Error('Not enough data for evaluation');
        }

        const predictions = await Promise.all(
            data.map(async d => {
                // Always explicitly specify dtype
                const inputTensor = this.tf.tensor2d([d.rawInput], [1, 15], 'float32');
                try {
                    const pred = this.model.predict(inputTensor) as tf.Tensor;
                    const value = (await pred.data())[0];
                    pred.dispose();
                    return value;
                } finally {
                    inputTensor.dispose();
                }
            })
        );

        const actual = data.map(d => d.actualOutcome);

        return {
            mae: this.calculateMAE(predictions, actual),
            rmse: this.calculateRMSE(predictions, actual),
            accuracy: this.calculateDirectionalAccuracy(predictions, actual)
        };
    }

    // Debug method to help isolate TensorFlow issues
    async debugModel(metrics: MarketMetrics): Promise<void> {
        console.log('=== DEBUG MODE ===');
        console.log('TensorFlow.js version:', this.tf.version.tfjs);

        // 1. Check input metrics
        console.log('\n1. Input Metrics:');

        // Check for required numeric fields
        const requiredNumericFields = [
            'currentPrice', 'priceVolatility', 'priceVelocity', 'trendStrength',
            'averageVolume', 'volumeVelocity', 'volumeConsistency', 'bidAskSpread',
            'turnoverRate', 'marketImpact', 'sma5', 'sma20', 'macdLine',
            'macdHistogram', 'liquidityScore'
        ];

        console.log('Checking required fields:');
        requiredNumericFields.forEach(field => {
            console.log(`${field}: ${typeof metrics[field as keyof MarketMetrics]} = ${metrics[field as keyof MarketMetrics]}`);
        });

        console.log(JSON.stringify(metrics, null, 2));

        try {
            // 2. Test feature extraction
            console.log('\n2. Feature Extraction:');
            const features = this.metricsToFeatures(metrics);
            console.log('Raw features:', features.raw);
            console.log('Features tensor shape:', features.tensor.shape);
            console.log('Features tensor dtype:', features.tensor.dtype);

            // 3. Test model inspection
            console.log('\n3. Model Structure:');
            this.model.summary();

            // 4. Test each layer individually
            console.log('\n4. Layer by Layer Test:');
            const inputTensor = this.tf.tensor2d(features.raw, [1, 15], 'float32');

            // Get all layers
            const layers = this.model.layers;
            console.log(`Model has ${layers.length} layers`);

            // Using a generic Tensor type to avoid type issues between layers
            let layerInput: tf.Tensor = inputTensor;
            for (let i = 0; i < layers.length; i++) {
                const layer = layers[i];
                console.log(`\nTesting layer ${i}: ${layer.name}`);
                console.log('Layer config:', layer.getConfig());

                try {
                    // Execute just this layer
                    const layerOutput = layer.apply(layerInput) as tf.Tensor;
                    console.log('Layer output shape:', layerOutput.shape);
                    console.log('Layer output dtype:', layerOutput.dtype);

                    // Clean up previous tensor if it's not the input tensor
                    if (i > 0) {
                        layerInput.dispose();
                    }

                    // Use this output as input to the next layer
                    layerInput = layerOutput;
                } catch (error) {
                    console.error(`Error in layer ${i} (${layer.name}):`, error);
                    throw error;
                }
            }

            // Clean up the last layer output
            if (layers.length > 0) {
                layerInput.dispose();

                // 5. Test full model prediction with logging
                console.log('\n5. Full Model Prediction:');
                const outputTensor = this.model.predict(inputTensor) as tf.Tensor;
                console.log('Output tensor shape:', outputTensor.shape);
                console.log('Output tensor dtype:', outputTensor.dtype);
                const outputValue = await outputTensor.data();
                console.log('Prediction value:', outputValue[0]);

                // Clean up
                console.log('\nCleaning up tensors...');
                inputTensor.dispose();
                features.tensor.dispose();
                outputTensor.dispose();

                console.log('=== DEBUG COMPLETE ===');
            }
        }catch (error) {
                console.error('DEBUG ERROR:', error);
                throw error;
            }
        }
    }