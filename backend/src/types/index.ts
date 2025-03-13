// src/types/index.ts
export interface TimeseriesData {
    timestamp: number;
    price: number;
    volume?: number;
}

export interface MarketMetrics {
    // Price Metrics
    currentPrice: number;
    priceVolatility: number;
    priceVelocity: number;
    trendStrength: number;

    // Volume Metrics
    averageVolume: number;
    volumeVelocity: number;
    volumeConsistency: number;

    // Trading Metrics
    bidAskSpread: number;
    turnoverRate: number;
    marketImpact: number;

    // Time Patterns
    hourlyPatterns: Record<number, number>;
    weekdayPatterns: Record<number, number>;

    // Technical Indicators
    sma5: number;
    sma20: number;
    macdLine: number;
    macdHistogram: number;

    // Market Conditions
    buyerCompetition: number;
    sellerCompetition: number;
    liquidityScore: number;

    // Flip Potential
    potentialProfit: number;
    riskScore: number;
}
export interface FlipPrediction {
    itemId: string;
    itemName: string;
    metrics: MarketMetrics;
    buyRecommendation: number;  // 0-1 score
    sellRecommendation: number; // 0-1 score
    predictedPriceChange: number;
    confidence: number;
}