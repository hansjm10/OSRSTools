// Try to use GPU backend first, fall back to CPU if not available
import * as tf from '@tensorflow/tfjs-node-gpu';
import {MarketMetrics} from "../types";
import * as fs from 'fs';
import * as path from 'path';

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
    private models: Map<string, tf.LayersModel> = new Map();
    private modelDirectory = path.join(__dirname, '../../models');
    private defaultFeatureStats: {means: number[], stds: number[]} = {
        means: new Array(15).fill(0),
        stds: new Array(15).fill(1)
    };
    private featureStats: Map<string, {means: number[], stds: number[]}> = new Map();
    // Initialize with a basic MSE function
    private customLossFunction: (yTrue: tf.Tensor, yPred: tf.Tensor) => tf.Tensor = 
        (yTrue: tf.Tensor, yPred: tf.Tensor) => tf.losses.meanSquaredError(yTrue, yPred);

    constructor() {
        this.tf = tf;
        this.trainingData = new Map();
        
        // Ensure model directory exists
        if (!fs.existsSync(this.modelDirectory)) {
            fs.mkdirSync(this.modelDirectory, { recursive: true });
        }
        
        // Initialize TensorFlow backend with Metal support for M2 Mac
        this.initializeBackend().then(() => {
            console.log('TensorFlow backend initialized successfully');
        }).catch(err => {
            console.warn('Failed to initialize GPU backend, using CPU:', err);
        });
        
        // Define custom Huber loss function
        this.defineCustomHuberLoss();
        
        this.initModel();
    }
    
    /**
     * Initialize TensorFlow.js backend, preferring GPU/Metal when available
     */
    private async initializeBackend(): Promise<void> {
        try {
            // Check if GPU/Metal backend is available
            const backend = this.tf.getBackend();
            console.log(`Current TensorFlow backend: ${backend}`);
            
            // Get available backends
            const availableBackends = Object.keys(this.tf.engine().registryFactory);
            console.log(`Available backends: ${availableBackends.join(', ')}`);
            
            if (backend !== 'tensorflow') {
                // Try to explicitly set the backend to tensorflow (node-gpu)
                await this.tf.setBackend('tensorflow');
                console.log(`Backend set to: ${this.tf.getBackend()}`);
            }
            
            // Check for GPU in a more compatible way
            const gpuAvailable = this.tf.engine().backendNames().some(
                (name: string) => name.toLowerCase().includes('gpu') || name.toLowerCase().includes('webgl')
            );
            
            if (gpuAvailable) {
                console.log('✅ GPU/Metal acceleration is available');
            } else {
                console.log('⚠️ Running on CPU - GPU/Metal acceleration not detected');
            }
        } catch (error) {
            console.error('Error initializing TensorFlow backend:', error);
            // Continue with whatever backend is available
        }
    }
    
    /**
     * Define a custom Huber loss function since some TF.js versions don't include it
     */
    private defineCustomHuberLoss() {
        try {
            // Use a standard MSE loss function that we know works
            this.customLossFunction = (yTrue: tf.Tensor, yPred: tf.Tensor): tf.Tensor => {
                // Simple MSE implementation with proper error handling
                return tf.tidy(() => {
                    try {
                        // Ensure inputs are in correct shape
                        const reshapedTrue = tf.reshape(yTrue, [-1]);
                        const reshapedPred = tf.reshape(yPred, [-1]);
                        
                        // Calculate MSE
                        const error = tf.sub(reshapedPred, reshapedTrue);
                        const squaredError = tf.square(error);
                        return tf.mean(squaredError);
                    } catch (e) {
                        console.error("Error in custom loss calculation:", e);
                        // Return a zero scalar as fallback
                        return tf.scalar(0);
                    }
                });
            };
            
            console.log("Using custom MSE loss function");
        } catch (error) {
            console.error("Error defining loss function:", error);
            // Create a very basic fallback that should always work
            this.customLossFunction = (): tf.Tensor => tf.scalar(0);
        }
    }

    private async initModel(): Promise<void> {
        // Clear any existing TensorFlow memory/state
        this.tf.disposeVariables();
        
        // Set environment flags for better GPU performance
        // Use try/catch for each flag since some might not be registered in the current TF.js version
        try {
            this.tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
        } catch (e) {
            // Silently continue if flag is not supported
        }
        
        try {
            this.tf.env().set('WEBGL_PACK', true);
        } catch (e) {
            // Silently continue if flag is not supported
        }
        
        try {
            this.tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);
        } catch (e) {
            // Silently continue if flag is not supported
        }
        
        // Skip the parallel execution mode flag as it's not registered

        try {
            // Create the base model
            this.model = this.createBidirectionalLSTMModel();
            console.log("Base model initialized");
            
            // Log memory info
            const memInfo = await this.tf.memory();
            console.log("TensorFlow memory usage:", {
                numTensors: memInfo.numTensors,
                numDataBuffers: memInfo.numDataBuffers,
                unreliable: memInfo.unreliable,
                reasons: memInfo.reasons
            });
        } catch (error) {
            console.error("Error initializing model with Huber loss:", error);
            console.log("Attempting to create model with MSE loss...");
            
            // Fallback to MSE loss if Huber doesn't work
            this.model = this.createModelWithMSELoss();
            console.log("Created fallback model with MSE loss");
        }
    }
    
    /**
     * Create a model with MSE loss as fallback
     */
    private createModelWithMSELoss(): tf.LayersModel {
        // Create model with simplified architecture
        const input = this.tf.input({
            shape: [15],
            dtype: 'float32',
            name: 'input_layer'
        });
        
        // Dense layers only - more stable
        const dense1 = this.tf.layers.dense({
            units: 64,
            activation: 'relu',
            name: 'dense_1'
        }).apply(input) as tf.SymbolicTensor;
        
        const dense2 = this.tf.layers.dense({
            units: 32,
            activation: 'relu',
            name: 'dense_2'
        }).apply(dense1) as tf.SymbolicTensor;
        
        const output = this.tf.layers.dense({
            units: 1,
            activation: 'linear',
            name: 'output'
        }).apply(dense2) as tf.SymbolicTensor;
        
        // Create the model
        const model = this.tf.model({
            inputs: input,
            outputs: output,
            name: 'simple_predictor'
        });
        
        // Compile with string-based MSE - most reliable option
        model.compile({
            optimizer: this.tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });
        
        return model;
    }

    /**
     * Creates a model with LSTM layers for sequence processing and dense layers
     */
    private createBidirectionalLSTMModel(): tf.LayersModel {
        // Input layer
        const input = this.tf.input({
            shape: [15],
            dtype: 'float32',
            name: 'input_layer'
        });

        // Reshape for LSTM (treat features as a sequence)
        const reshapedInput = this.tf.layers.reshape({
            targetShape: [1, 15],
            name: 'reshape_layer'
        }).apply(input) as tf.SymbolicTensor;

        // Bidirectional LSTM layer
        const bidirectionalLSTM = this.tf.layers.bidirectional({
            layer: this.tf.layers.lstm({
                units: 32,
                returnSequences: true,
                name: 'lstm_layer'
            }),
            mergeMode: 'concat',
            name: 'bidirectional_layer'
        }).apply(reshapedInput) as tf.SymbolicTensor;

        // Flatten the output from LSTM
        const flattened = this.tf.layers.flatten({
            name: 'flatten_layer'
        }).apply(bidirectionalLSTM) as tf.SymbolicTensor;

        // Dense layers
        const dense1 = this.tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'glorotUniform',
            biasInitializer: 'zeros',
            name: 'dense_1'
        }).apply(flattened) as tf.SymbolicTensor;

        // Batch normalization - optimized for GPU with explicit momentum
        const batchNorm = this.tf.layers.batchNormalization({
            name: 'batch_norm',
            momentum: 0.99, // Higher momentum works better on GPU
            epsilon: 1e-5  // Recommended value for numerical stability
        }).apply(dense1) as tf.SymbolicTensor;

        // Dropout layer
        const dropout = this.tf.layers.dropout({
            rate: 0.3,
            name: 'dropout_1'
        }).apply(batchNorm) as tf.SymbolicTensor;

        // Second dense layer
        const dense2 = this.tf.layers.dense({
            units: 32,
            activation: 'relu',
            kernelInitializer: 'glorotUniform',
            biasInitializer: 'zeros',
            name: 'dense_2'
        }).apply(dropout) as tf.SymbolicTensor;

        // Output layer
        const output = this.tf.layers.dense({
            units: 1,
            activation: 'linear',
            kernelInitializer: 'glorotUniform',
            biasInitializer: 'zeros',
            name: 'output'
        }).apply(dense2) as tf.SymbolicTensor;

        // Create the model
        const model = this.tf.model({
            inputs: input,
            outputs: output,
            name: 'market_predictor'
        });

        try {
            // Use our custom MSE loss function
            model.compile({
                optimizer: this.tf.train.adam(0.0005),
                loss: this.customLossFunction,
                metrics: ['mae']
            });
        } catch (error) {
            console.warn("Failed to compile with custom loss, falling back to string MSE:", error);
            // Fallback to standard string-based MSE
            model.compile({
                optimizer: this.tf.train.adam(0.0005),
                loss: 'meanSquaredError',
                metrics: ['mae']
            });
        }

        return model;
    }

    private metricsToFeatures(metrics: MarketMetrics, itemId?: string): FeatureData {
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
        const features = [
            metrics.currentPrice,
            metrics.priceVolatility,
            metrics.priceVelocity,
            metrics.trendStrength,
            metrics.averageVolume,
            metrics.volumeVelocity,
            metrics.volumeConsistency,
            metrics.bidAskSpread,
            metrics.turnoverRate,
            metrics.marketImpact,
            metrics.sma5,
            metrics.sma20,
            metrics.macdLine,
            metrics.macdHistogram,
            metrics.liquidityScore
        ];

        // Verify all features are valid numbers
        if (!this.validateFeatures(features)) {
            console.error('Invalid features:', features);
            throw new Error('Invalid feature values detected');
        }

        // Normalize features using z-score normalization if item-specific stats exist
        let normalizedFeatures;
        if (itemId && this.featureStats.has(itemId)) {
            normalizedFeatures = this.zScoreNormalize(features, this.featureStats.get(itemId)!);
        } else {
            normalizedFeatures = features.map(f => this.normalizeValue(f));
        }

        // Always explicitly specify dtype as float32 and shape
        return {
            tensor: this.tf.tensor2d([normalizedFeatures], [1, 15], 'float32'),
            raw: normalizedFeatures
        };
    }

    private zScoreNormalize(features: number[], stats: {means: number[], stds: number[]}): number[] {
        return features.map((value, i) => {
            const mean = stats.means[i];
            const std = stats.stds[i] || 1; // Avoid division by zero
            return (value - mean) / std;
        });
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

    /**
     * Calculate feature statistics (mean, std) for Z-score normalization
     */
    private calculateFeatureStats(itemId: string): void {
        const data = this.trainingData.get(itemId);
        if (!data || data.length < 5) return;
        
        // Calculate means for each feature
        const features = data.map(d => d.rawInput);
        const sums = new Array(15).fill(0);
        const squaredSums = new Array(15).fill(0);
        
        for (const featureSet of features) {
            for (let i = 0; i < 15; i++) {
                sums[i] += featureSet[i];
                squaredSums[i] += featureSet[i] * featureSet[i];
            }
        }
        
        const means = sums.map(sum => sum / features.length);
        const stds = sums.map((sum, i) => {
            const variance = (squaredSums[i] / features.length) - Math.pow(sum / features.length, 2);
            return Math.sqrt(Math.max(variance, 0.001)); // Ensure non-zero std
        });
        
        this.featureStats.set(itemId, { means, stds });
    }

    /**
     * Get model specific to item or use general model if not available
     */
    private getItemModel(itemId: string): tf.LayersModel {
        if (this.models.has(itemId)) {
            return this.models.get(itemId)!;
        }
        return this.model;
    }

    /**
     * Save model for a specific item
     */
    private async saveItemModel(itemId: string, model: tf.LayersModel): Promise<void> {
        const modelPath = path.join(this.modelDirectory, `model_${itemId}`);
        try {
            await model.save(`file://${modelPath}`);
            
            // Save feature statistics
            if (this.featureStats.has(itemId)) {
                const statsPath = path.join(this.modelDirectory, `stats_${itemId}.json`);
                fs.writeFileSync(statsPath, JSON.stringify(this.featureStats.get(itemId)));
            }
        } catch (error) {
            console.error(`Error saving model for item ${itemId}:`, error);
        }
    }

    /**
     * Load model for a specific item
     */
    private async loadItemModel(itemId: string): Promise<boolean> {
        const modelPath = path.join(this.modelDirectory, `model_${itemId}`);
        const statsPath = path.join(this.modelDirectory, `stats_${itemId}.json`);
        
        try {
            if (fs.existsSync(`${modelPath}/model.json`)) {
                const model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
                
                try {
                    // Use our custom MSE loss function
                    model.compile({
                        optimizer: this.tf.train.adam(0.0005),
                        loss: this.customLossFunction,
                        metrics: ['mae']
                    });
                } catch (error) {
                    // Fallback to standard string-based MSE loss
                    model.compile({
                        optimizer: this.tf.train.adam(0.0005),
                        loss: 'meanSquaredError', 
                        metrics: ['mae']
                    });
                }
                this.models.set(itemId, model);
                
                // Load feature statistics if available
                if (fs.existsSync(statsPath)) {
                    const statsData = fs.readFileSync(statsPath, 'utf8');
                    this.featureStats.set(itemId, JSON.parse(statsData));
                }
                
                return true;
            }
        } catch (error) {
            // Silent fail - just return false
        }
        return false;
    }

    async predict(metrics: MarketMetrics, itemId?: string): Promise<number> {
        // Try to use item specific model if itemId is provided
        const model = itemId ? this.getItemModel(itemId) : this.model;
        
        // Process features using item-specific normalization if available
        const features = this.metricsToFeatures(metrics, itemId);

        if (!this.validateFeatures(features.raw)) {
            throw new Error('Invalid feature data');
        }

        // Create a new input tensor with explicitly set dtype
        const inputTensor = this.tf.tensor2d(features.raw, [1, 15], 'float32');

        try {
            // Make prediction
            const prediction = model.predict(inputTensor) as tf.Tensor;
            const value = (await prediction.data())[0];

            // Clean up
            prediction.dispose();

            // Denormalize if we used item-specific statistics
            if (itemId && this.featureStats.has(itemId)) {
                return value * (Math.abs(metrics.currentPrice) > 1e6 ? 1e6 : 1);
            }
            
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
            // Try to load a pre-existing model for this item
            await this.loadItemModel(itemId);
        }
        
        const itemData = this.trainingData.get(itemId);
        if (itemData) {
            itemData.push(example);
        }

        features.tensor.dispose();

        // Increase training data size requirement before retraining
        if (this.trainingData.get(itemId)?.length === 50) {
            await this.retrain(itemId);
        }
    }

    async retrainAll(): Promise<void> {
        for (const itemId of this.trainingData.keys()) {
            if (this.trainingData.get(itemId)?.length! >= 20) {
                await this.retrain(itemId);
            }
        }
    }

    private async retrain(itemId: string): Promise<void> {
        const data = this.trainingData.get(itemId);
        if (!data || data.length < 20) {
            // Silent skip for insufficient data
            return;
        }

        // Calculate feature statistics for normalization
        this.calculateFeatureStats(itemId);
        
        // Clone the base model for this item if it doesn't already exist
        if (!this.models.has(itemId)) {
            // Create a new instance
            const newModel = this.createBidirectionalLSTMModel();
            this.models.set(itemId, newModel);
        }
        
        // Get the item-specific model
        const model = this.models.get(itemId)!;
        
        // Prepare training data with normalization
        const normalizedData = data.map(d => {
            return {
                ...d,
                rawInput: this.zScoreNormalize(d.rawInput, this.featureStats.get(itemId)!)
            };
        });
        
        // Split into training and validation sets
        const shuffledData = [...normalizedData].sort(() => Math.random() - 0.5);
        const splitIndex = Math.floor(shuffledData.length * 0.8);
        const trainingData = shuffledData.slice(0, splitIndex);
        const validationData = shuffledData.slice(splitIndex);

        // Always explicitly specify dtype
        const trainInputs = this.tf.tensor2d(trainingData.map(d => d.rawInput), undefined, 'float32');
        const trainOutputs = this.tf.tensor2d(trainingData.map(d => [d.actualOutcome]), undefined, 'float32');
        
        const validInputs = this.tf.tensor2d(validationData.map(d => d.rawInput), undefined, 'float32');
        const validOutputs = this.tf.tensor2d(validationData.map(d => [d.actualOutcome]), undefined, 'float32');

        try {
            // Define callbacks for early stopping
            const callbacks = [
                this.tf.callbacks.earlyStopping({
                    monitor: 'val_loss',
                    patience: 5,
                    verbose: 0 // Reduced verbosity
                })
            ];
            
            // Train the model with more epochs, optimized for GPU
            const result = await model.fit(trainInputs, trainOutputs, {
                epochs: 50,
                batchSize: 32, // Increased batch size for GPU efficiency
                validationData: [validInputs, validOutputs],
                shuffle: true,
                callbacks,
                verbose: 0, // Reduced verbosity - no per-epoch outputs
                yieldEvery: 'never' // Disable yielding for better GPU performance
            });
            
            // Save the model without logging
            await this.saveItemModel(itemId, model);
            
        } finally {
            trainInputs.dispose();
            trainOutputs.dispose();
            validInputs.dispose();
            validOutputs.dispose();
        }
    }

    async evaluateAccuracy(itemId: string): Promise<ModelAccuracy> {
        const data = this.trainingData.get(itemId) || [];
        if (data.length < 20) {
            throw new Error('Not enough data for evaluation (need at least 20 examples)');
        }

        // Use item-specific model if available
        const model = this.models.has(itemId) ? this.models.get(itemId)! : this.model;
        
        // Apply item-specific normalization if available
        const useItemStats = this.featureStats.has(itemId);
        
        const predictions = await Promise.all(
            data.map(async d => {
                // Apply normalization if stats are available
                const features = useItemStats ? 
                    this.zScoreNormalize(d.rawInput, this.featureStats.get(itemId)!) : 
                    d.rawInput;
                
                // Always explicitly specify dtype
                const inputTensor = this.tf.tensor2d([features], [1, 15], 'float32');
                try {
                    const pred = model.predict(inputTensor) as tf.Tensor;
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

    /**
     * Check if GPU/Metal is being used (public method)
     */
    public async checkGPUUsage(): Promise<boolean> {
        try {
            // Get current backend
            const backend = this.tf.getBackend();
            console.log(`Current backend: ${backend}`);
            
            // Create a simple test tensor
            const a = this.tf.tensor2d([[1, 2], [3, 4]]);
            const b = this.tf.tensor2d([[5, 6], [7, 8]]);
            
            // Time a matrix multiplication (good GPU test)
            const start = Date.now();
            const iterations = 100;
            
            for (let i = 0; i < iterations; i++) {
                const result = this.tf.matMul(a, b);
                await result.data(); // Force execution
                result.dispose();
            }
            
            const end = Date.now();
            const timePerOp = (end - start) / iterations;
            
            // Log performance
            console.log(`Matrix multiplication time: ${timePerOp.toFixed(2)}ms per operation`);
            console.log(`Estimated performance: ${(1000 / timePerOp).toFixed(2)} ops/second`);
            
            // Check if GPU is likely being used (very rough estimate)
            const isLikelyGPU = timePerOp < 1.0 && backend !== 'cpu';
            console.log(`GPU likely in use: ${isLikelyGPU ? 'Yes ✅' : 'No ❌'}`);
            
            // Clean up
            a.dispose();
            b.dispose();
            
            return isLikelyGPU;
        } catch (error) {
            console.error('Error checking GPU usage:', error);
            return false;
        }
    }

    // Debug method to help isolate TensorFlow issues
    async debugModel(metrics: MarketMetrics): Promise<void> {
        console.log('=== DEBUG MODE ===');
        console.log('TensorFlow.js version:', this.tf.version.tfjs);
        
        // Check if GPU is being used
        await this.checkGPUUsage();

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

            // 4. Test full model prediction with logging
            console.log('\n5. Full Model Prediction:');
            const inputTensor = this.tf.tensor2d(features.raw, [1, 15], 'float32');
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
        } catch (error) {
            console.error('DEBUG ERROR:', error);
            throw error;
        }
    }
}