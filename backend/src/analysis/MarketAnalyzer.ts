// src/analysis/MarketAnalyzer.ts

import {MarketMetrics, TimeseriesData} from "../types";

export class MarketAnalyzer {
    constructor(private dataService: any) {}

    async analyzeItem(itemId: string): Promise<MarketMetrics> {
        const [history, bulkData] = await Promise.all([
            this.dataService.getItemHistory(itemId),
            this.dataService.getBulkData()
        ]);

        const item = bulkData[itemId];
        return this.calculateMetrics(history, item);
    }

    private calculateMetrics(history: TimeseriesData[], item: any): MarketMetrics {
        const prices = history.map(h => h.price);
        const volumes = history.map(h => h.volume || 0);

        return {
            currentPrice: prices[prices.length - 1],
            priceVolatility: this.calculateVolatility(prices),
            priceVelocity: this.calculateVelocity(prices),
            trendStrength: this.calculateTrendStrength(prices),

            averageVolume: this.calculateMean(volumes),
            volumeVelocity: this.calculateVelocity(volumes),
            volumeConsistency: this.calculateConsistency(volumes),

            bidAskSpread: this.estimateBidAskSpread(prices),
            turnoverRate: this.calculateTurnoverRate(volumes, item.limit),
            marketImpact: this.calculateMarketImpact(history),

            hourlyPatterns: this.calculateHourlyPatterns(history),
            weekdayPatterns: this.calculateWeekdayPatterns(history),

            sma5: this.calculateSMA(prices, 5),
            sma20: this.calculateSMA(prices, 20),
            macdLine: this.calculateMACD(prices).macdLine,
            macdHistogram: this.calculateMACD(prices).histogram,

            buyerCompetition: this.calculateCompetition(history, 'buy'),
            sellerCompetition: this.calculateCompetition(history, 'sell'),
            liquidityScore: this.calculateLiquidityScore(history, item),

            potentialProfit: this.calculatePotentialProfit(history, item),
            riskScore: this.calculateRiskScore(history, item)
        };
    }

    private calculateMean(values: number[]): number {
        return values.reduce((sum, val) => sum + val, 0) / values.length;
    }

    private calculateVolatility(values: number[]): number {
        const mean = this.calculateMean(values);
        const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
        return Math.sqrt(this.calculateMean(squaredDiffs));
    }

    private calculateVelocity(values: number[]): number {
        const changes = values.slice(1).map((val, i) => val - values[i]);
        return this.calculateMean(changes);
    }

    private calculateConsistency(values: number[]): number {
        const mean = this.calculateMean(values);
        const volatility = this.calculateVolatility(values);
        return 1 - (volatility / mean); // Higher is more consistent
    }

    private calculateTrendStrength(prices: number[]): number {
        const changes = prices.slice(1).map((price, i) => price - prices[i]);
        const positiveChanges = changes.filter(change => change > 0).length;
        return positiveChanges / changes.length;
    }

    private estimateBidAskSpread(prices: number[]): number {
        const volatility = this.calculateVolatility(prices);
        const currentPrice = prices[prices.length - 1];
        return (volatility / currentPrice) * 100; // As percentage of price
    }

    private calculateTurnoverRate(volumes: number[], limit: number): number {
        const averageVolume = this.calculateMean(volumes);
        return limit ? averageVolume / limit : 0;
    }

    private calculateMarketImpact(history: TimeseriesData[]): number {
        const priceChanges = history.slice(1).map((data, i) => {
            const priceChange = Math.abs(data.price - history[i].price);
            const volume = data.volume || 1;
            return priceChange / volume;
        });
        return this.calculateMean(priceChanges);
    }

    private calculateHourlyPatterns(history: TimeseriesData[]): Record<number, number> {
        const hourlyPrices: Record<number, number[]> = {};

        history.forEach(data => {
            const hour = new Date(data.timestamp * 1000).getUTCHours();
            if (!hourlyPrices[hour]) hourlyPrices[hour] = [];
            hourlyPrices[hour].push(data.price);
        });

        return Object.fromEntries(
            Object.entries(hourlyPrices).map(([hour, prices]) => [
                hour,
                this.calculateMean(prices)
            ])
        );
    }

    private calculateWeekdayPatterns(history: TimeseriesData[]): Record<number, number> {
        const weekdayPrices: Record<number, number[]> = {};

        history.forEach(data => {
            const weekday = new Date(data.timestamp * 1000).getUTCDay();
            if (!weekdayPrices[weekday]) weekdayPrices[weekday] = [];
            weekdayPrices[weekday].push(data.price);
        });

        return Object.fromEntries(
            Object.entries(weekdayPrices).map(([weekday, prices]) => [
                weekday,
                this.calculateMean(prices)
            ])
        );
    }

    private calculateSMA(values: number[], period: number): number {
        if (values.length < period) return values[values.length - 1];
        return this.calculateMean(values.slice(-period));
    }

    private calculateMACD(prices: number[]): { macdLine: number; histogram: number } {
        const ema12 = this.calculateEMA(prices, 12);
        const ema26 = this.calculateEMA(prices, 26);
        const macdLine = ema12 - ema26;
        const signalLine = this.calculateEMA([macdLine], 9);
        const histogram = macdLine - signalLine;

        return { macdLine, histogram };
    }

    private calculateEMA(values: number[], period: number): number {
        const k = 2 / (period + 1);
        return values.reduce((ema, price) =>
                price * k + ema * (1 - k),
            values[0]
        );
    }

    private calculateCompetition(history: TimeseriesData[], type: 'buy' | 'sell'): number {
        const priceChanges = history.slice(1).map((data, i) => ({
            change: data.price - history[i].price,
            volume: data.volume || 0
        }));

        const relevantChanges = type === 'buy'
            ? priceChanges.filter(change => change.change > 0)
            : priceChanges.filter(change => change.change < 0);

        if (relevantChanges.length === 0) return 0;

        const volumeWeightedChanges = relevantChanges.map(
            change => Math.abs(change.change) * change.volume
        );

        return this.calculateMean(volumeWeightedChanges);
    }

    private calculateLiquidityScore(history: TimeseriesData[], item: any): number {
        const volumeScore = this.calculateConsistency(history.map(h => h.volume || 0));
        const spreadScore = 1 - this.estimateBidAskSpread(history.map(h => h.price)) / 100;
        const turnoverScore = this.calculateTurnoverRate(
            history.map(h => h.volume || 0),
            item.limit
        );

        return (volumeScore + spreadScore + turnoverScore) / 3;
    }

    private calculatePotentialProfit(history: TimeseriesData[], item: any): number {
        const recentPrices = history.slice(-24); // Last day of data
        const minPrice = Math.min(...recentPrices.map(h => h.price));
        const maxPrice = Math.max(...recentPrices.map(h => h.price));
        const currentPrice = history[history.length - 1].price;

        const potentialBuyProfit = maxPrice - currentPrice;
        const potentialSellProfit = currentPrice - minPrice;

        return Math.max(potentialBuyProfit, potentialSellProfit);
    }

    private calculateRiskScore(history: TimeseriesData[], item: any): number {
        const volatilityRisk = this.calculateVolatility(history.map(h => h.price));
        const liquidityRisk = 1 - this.calculateLiquidityScore(history, item);
        const competitionRisk = Math.max(
            this.calculateCompetition(history, 'buy'),
            this.calculateCompetition(history, 'sell')
        );

        // Normalize and combine risks
        const normalizedVolatilityRisk = Math.min(volatilityRisk / item.price, 1);
        const normalizedCompetitionRisk = Math.min(competitionRisk / item.price, 1);

        return (
            normalizedVolatilityRisk * 0.4 +
            liquidityRisk * 0.3 +
            normalizedCompetitionRisk * 0.3
        );
    }
}