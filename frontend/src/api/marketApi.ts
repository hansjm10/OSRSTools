// src/api/marketApi.ts
import axios from 'axios';

// Base URL for the backend API
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

// Create an axios instance with default config
const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// API functions for the market dashboard
export const marketApi = {
    // Get system status and performance metrics
    getSystemStatus: async () => {
        try {
            const response = await apiClient.get('/system/status');
            return response.data;
        } catch (error) {
            console.error('Error fetching system status:', error);
            throw error;
        }
    },

    // Get current buy/sell recommendations
    getRecommendations: async () => {
        try {
            const response = await apiClient.get('/recommendations');
            return response.data;
        } catch (error) {
            console.error('Error fetching recommendations:', error);
            throw error;
        }
    },

    // Get model accuracy data for analytics
    getAccuracyData: async () => {
        try {
            const response = await apiClient.get('/analytics/accuracy');
            return response.data;
        } catch (error) {
            console.error('Error fetching accuracy data:', error);
            throw error;
        }
    },

    // Get price history data for charts
    getPriceHistory: async () => {
        try {
            const response = await apiClient.get('/analytics/price-history');
            return response.data;
        } catch (error) {
            console.error('Error fetching price history:', error);
            throw error;
        }
    },

    // Get detailed data for a specific item
    getItemDetails: async (itemId: number) => {
        try {
            const response = await apiClient.get(`/items/${itemId}`);
            return response.data;
        } catch (error) {
            console.error(`Error fetching details for item ${itemId}:`, error);
            throw error;
        }
    },

    // Manually trigger a system update
    triggerUpdate: async () => {
        try {
            const response = await apiClient.post('/system/update');
            return response.data;
        } catch (error) {
            console.error('Error triggering system update:', error);
            throw error;
        }
    }
};

export default marketApi;