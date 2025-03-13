// src/components/MarketDashboard.tsx
import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, Clock, DollarSign, Percent, BarChart2, RefreshCw, AlertCircle } from 'lucide-react';
import { marketApi } from '../api/marketApi';
import '../styles/dashboard.css'; // Import the CSS

// Define the structure of your API responses
interface SystemStatusResponse {
    status: string;
    performance: {
        dailyPredictions: number;
        accuracyRate: number;
        avgConfidence: number;
        totalItemsTracked: number;
        trainingTime: number;
        lastUpdate: string;
    };
    itemsTracked: number;
}

interface ItemMetrics {
    currentPrice: number;
    averageVolume: number;
    liquidityScore?: number;
    [key: string]: any; // For any other properties
}

interface ItemPrediction {
    itemId: string;
    itemName: string;
    predictedPriceChange: number;
    confidence: number;
    metrics: ItemMetrics;
    buyRecommendation?: number;
    sellRecommendation?: number;
    quantity?: number; // New field for recommended quantity
}

interface ModelAccuracy {
    itemId: string;
    itemName: string;
    accuracy: number;
    mae: number;
    rmse: number;
}

interface PriceHistoryPoint {
    date: string;
    [key: string]: number | string; // Item prices keyed by item ID or name
}

// Define the structure of your combined state
interface MarketData {
    systemStatus: SystemStatusResponse | null;
    recommendations: {
        bestBuys: ItemPrediction[];
        bestSells: ItemPrediction[];
    };
    priceHistory: PriceHistoryPoint[];
    modelAccuracies: ModelAccuracy[];
}

// Component prop types
interface DashboardCardProps {
    title: string;
    value: string | number;
    icon: React.ComponentType<any>;
    color: string;
    subvalue?: string;
}

interface ItemTableRowProps {
    item: ItemPrediction;
    type: 'buy' | 'sell';
}

interface TradeTabProps {
    type: 'buy' | 'sell';
    items: ItemPrediction[];
}

// Format large numbers with K, M suffix
const formatNumber = (num: number | undefined | null): string => {
    if (!num && num !== 0) return '-';
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
};

// Format date strings to relative time
const formatRelativeTime = (dateString: string | undefined | null): string => {
    if (!dateString) return 'Unknown';

    const date = new Date(dateString);
    const now = new Date();
    const diffSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (diffSeconds < 60) return 'Just now';
    if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m ago`;
    if (diffSeconds < 86400) return `${Math.floor(diffSeconds / 3600)}h ago`;
    return `${Math.floor(diffSeconds / 86400)}d ago`;
};

const ItemTableRow: React.FC<ItemTableRowProps> = ({ item, type }) => {
    const isProfitable = type === 'buy' ?
        (item.predictedPriceChange > 0) :
        (item.predictedPriceChange < 0);

    const ROI = ((Math.abs(item.predictedPriceChange) / item.metrics.currentPrice) * 100).toFixed(1);
    const potentialProfit = type === 'buy'
        ? item.metrics.currentPrice + item.predictedPriceChange
        : item.metrics.currentPrice - Math.abs(item.predictedPriceChange);

    // Calculate total profit/loss with quantity
    const totalValue = item.quantity 
        ? Math.abs(item.predictedPriceChange * (item.quantity || 0)) 
        : 0;
    
    return (
        <tr>
            <td>
                <div className="item-name">{item.itemName}</div>
                <div className="item-id">ID: {item.itemId}</div>
            </td>
            <td className="text-right">{item.metrics.currentPrice.toLocaleString()} gp</td>
            <td className={`text-right ${isProfitable ? 'text-green' : 'text-red'}`}>
                {type === 'buy' ? '+' : ''}{item.predictedPriceChange.toFixed(1)} gp
            </td>
            <td className="text-right">
                {potentialProfit.toLocaleString()} gp
            </td>
            <td className="text-right">{ROI}%</td>
            <td className="text-right">{item.confidence ? (item.confidence * 100).toFixed(1) : '-'}%</td>
            <td className="text-right">
                <div>{formatNumber(item.quantity || 0)}</div>
                <div className="text-xs text-green">
                    {totalValue > 0 ? `${formatNumber(totalValue)} gp` : ''}
                </div>
            </td>
            <td className="text-right hidden md:table-cell">{formatNumber(item.metrics.averageVolume)}</td>
            <td className="text-right hidden lg:table-cell">{item.metrics.liquidityScore?.toFixed(1) || '-'}</td>
        </tr>
    );
};

const DashboardCard: React.FC<DashboardCardProps> = ({ title, value, icon, color, subvalue }) => {
    const Icon = icon;

    return (
        <div className={`stat-card ${color.replace('bg-', '')}`}>
            <div className="stat-card-content">
                <div className="stat-card-text">
                    <h3 className="stat-card-title">{title}</h3>
                    <div>
                        <span className="stat-card-value">{value}</span>
                        {subvalue && <span className="stat-card-subvalue">{subvalue}</span>}
                    </div>
                </div>
                <div className={`stat-card-icon ${color}`}>
                    <Icon size={20} />
                </div>
            </div>
        </div>
    );
};

const TradeTab: React.FC<TradeTabProps> = ({ type, items }) => {
    if (!items || items.length === 0) {
        return (
            <div className="empty-state">
                <AlertCircle size={32} />
                <p>No {type} recommendations available at this time.</p>
            </div>
        );
    }

    return (
        <div className="table-container">
            <table className="data-table">
                <thead>
                <tr>
                    <th>Item</th>
                    <th className="text-right">Current</th>
                    <th className="text-right">Predicted Δ</th>
                    <th className="text-right">Target Price</th>
                    <th className="text-right">ROI</th>
                    <th className="text-right">Confidence</th>
                    <th className="text-right">Quantity</th>
                    <th className="text-right hidden md:table-cell">Volume</th>
                    <th className="text-right hidden lg:table-cell">Liquidity</th>
                </tr>
                </thead>
                <tbody>
                {items.map((item) => (
                    <ItemTableRow key={item.itemId} item={item} type={type} />
                ))}
                </tbody>
            </table>
        </div>
    );
};

const MarketDashboard: React.FC = () => {
    const [marketData, setMarketData] = useState<MarketData>({
        systemStatus: null,
        recommendations: { bestBuys: [], bestSells: [] },
        priceHistory: [],
        modelAccuracies: []
    });

    const [activeTab, setActiveTab] = useState('buy');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Load all data from API
    const loadAllData = async () => {
        setLoading(true);
        setError(null);

        try {
            // Fetch all required data in parallel
            const [statusData, recommendationsData, accuracyData, priceHistoryData] = await Promise.all([
                marketApi.getSystemStatus(),
                marketApi.getRecommendations(),
                marketApi.getAccuracyData(),
                marketApi.getPriceHistory()
            ]);

            setMarketData({
                systemStatus: statusData,
                recommendations: recommendationsData.recommendations,
                priceHistory: priceHistoryData.priceHistory || [],
                modelAccuracies: accuracyData.accuracyData || []
            });

            setLoading(false);
        } catch (err) {
            console.error('Error loading market data:', err);
            setError('Failed to load market data. Please try again later.');
            setLoading(false);
        }
    };

    // Refresh data
    const handleRefresh = async () => {
        try {
            setLoading(true);
            await marketApi.triggerUpdate();
            await loadAllData();
        } catch (err) {
            console.error('Error refreshing data:', err);
            setError('Failed to refresh market data. Please try again later.');
            setLoading(false);
        }
    };

    // Initial data load
    useEffect(() => {
        loadAllData();

        // Set up auto-refresh interval (every 5 minutes)
        const intervalId = setInterval(() => {
            loadAllData();
        }, 5 * 60 * 1000);

        // Clean up interval on component unmount
        return () => clearInterval(intervalId);
    }, []);

    if (loading && !marketData.systemStatus) {
        return (
            <div className="dashboard-container">
                <div className="error-container">
                    <div className="text-center">
                        <div className="spinner"></div>
                        <p className="mt-2 text-gray-600">Loading market data...</p>
                    </div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="dashboard-container">
                <div className="error-container">
                    <div className="error-card">
                        <AlertCircle size={48} className="error-icon" />
                        <h2 className="error-title">Connection Error</h2>
                        <p className="error-message">{error}</p>
                        <button
                            onClick={loadAllData}
                            className="retry-button"
                        >
                            Try Again
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    const { systemStatus, recommendations, priceHistory, modelAccuracies } = marketData;
    const performance = systemStatus?.performance;

    return (
        <div className="dashboard-container">
            <div className="dashboard-content">
                <div className="dashboard-header">
                    <h1 className="dashboard-title">OSRS Market Trading Dashboard</h1>
                    <p className="dashboard-subtitle">Real-time market predictions and trading recommendations</p>
                </div>

                {/* Stats Cards */}
                <div className="stats-grid">
                    <DashboardCard
                        title="Prediction Accuracy"
                        value={`${performance?.accuracyRate ? (performance.accuracyRate * 100).toFixed(1) : '-'}%`}
                        icon={TrendingUp}
                        color="green"
                    />
                    <DashboardCard
                        title="Items Tracked"
                        value={performance?.totalItemsTracked || '-'}
                        icon={BarChart2}
                        color="blue"
                    />
                    <DashboardCard
                        title="Avg. Confidence"
                        value={`${performance?.avgConfidence ? (performance.avgConfidence * 100).toFixed(1) : '-'}%`}
                        icon={Percent}
                        color="purple"
                    />
                    <DashboardCard
                        title="ML Model Type"
                        value={"Bidirectional LSTM"}
                        icon={DollarSign}
                        color="pink"
                        subvalue="Item-specific models"
                    />
                    <DashboardCard
                        title="Last Update"
                        value={performance?.lastUpdate ? formatRelativeTime(performance.lastUpdate) : 'Unknown'}
                        icon={Clock}
                        color="indigo"
                        subvalue="auto-updates every 30min"
                    />
                </div>

                {/* Tabs */}
                <div className="tab-container">
                    <div className="tab-header">
                        <nav className="tab-nav">
                            <button
                                onClick={() => setActiveTab('buy')}
                                className={`tab-button ${activeTab === 'buy' ? 'active' : ''}`}
                            >
                                <TrendingUp size={16} />
                                Buy Recommendations
                            </button>
                            <button
                                onClick={() => setActiveTab('sell')}
                                className={`tab-button ${activeTab === 'sell' ? 'active' : ''}`}
                            >
                                <TrendingDown size={16} />
                                Sell Recommendations
                            </button>
                            <button
                                onClick={() => setActiveTab('analytics')}
                                className={`tab-button ${activeTab === 'analytics' ? 'active' : ''}`}
                            >
                                <BarChart2 size={16} />
                                Analytics
                            </button>
                        </nav>
                    </div>

                    <div className="tab-content">
                        {loading && (
                            <div className="tab-content-loader">
                                <div className="spinner"></div>
                            </div>
                        )}

                        {activeTab === 'buy' && (
                            <TradeTab type="buy" items={recommendations.bestBuys} />
                        )}
                        {activeTab === 'sell' && (
                            <TradeTab type="sell" items={recommendations.bestSells} />
                        )}
                        {activeTab === 'analytics' && (
                            <div className="chart-grid">
                                <div className="chart-card">
                                    <h3 className="chart-title">Price History - Top Items</h3>

                                    {priceHistory && priceHistory.length > 0 ? (
                                        <div className="chart-container">
                                            <ResponsiveContainer width="100%" height="100%">
                                                <LineChart
                                                    data={priceHistory}
                                                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                                                >
                                                    <CartesianGrid strokeDasharray="3 3" />
                                                    <XAxis dataKey="date" />
                                                    <YAxis />
                                                    <Tooltip />
                                                    <Legend />
                                                    {Object.keys(priceHistory[0])
                                                        .filter(key => key !== 'date')
                                                        .slice(0, 5) // Show only top 5 items
                                                        .map((itemKey, index) => {
                                                            // Generate different colors for each line
                                                            const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088fe'];
                                                            return (
                                                                <Line
                                                                    key={itemKey}
                                                                    type="monotone"
                                                                    dataKey={itemKey}
                                                                    name={itemKey}
                                                                    stroke={colors[index % colors.length]}
                                                                />
                                                            );
                                                        })}
                                                </LineChart>
                                            </ResponsiveContainer>
                                        </div>
                                    ) : (
                                        <div className="no-data">
                                            No price history data available yet
                                        </div>
                                    )}
                                </div>
                                <div className="chart-card">
                                    <h3 className="chart-title">Model Accuracy by Item</h3>

                                    {modelAccuracies && modelAccuracies.length > 0 ? (
                                        <div className="chart-container">
                                            <ResponsiveContainer width="100%" height="100%">
                                                <BarChart
                                                    data={modelAccuracies}
                                                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                                                >
                                                    <CartesianGrid strokeDasharray="3 3" />
                                                    <XAxis dataKey="itemName" tick={{ fontSize: 10 }} />
                                                    <YAxis domain={[0, 1]} />
                                                    <Tooltip formatter={(value: any) => `${(Number(value) * 100).toFixed(1)}%`} />
                                                    <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" />
                                                </BarChart>
                                            </ResponsiveContainer>
                                        </div>
                                    ) : (
                                        <div className="no-data">
                                            No model accuracy data available yet
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Action buttons */}
                <div className="action-buttons">
                    <button
                        onClick={handleRefresh}
                        disabled={loading}
                        className="button button-primary"
                    >
                        <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
                        Refresh Data
                    </button>
                    <button className="button button-success">
                        <DollarSign size={16} />
                        Start Trading Bot
                    </button>
                </div>
            </div>
        </div>
    );
};

export default MarketDashboard;