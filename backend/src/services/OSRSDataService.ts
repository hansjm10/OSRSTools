// src/services/OSRSDataService.ts

import { TimeseriesData } from "../types";
import fetch from 'node-fetch';

export interface BulkDataResponse {
    [key: string]: {
        name: string;
        limit: number;
        members: boolean;
        value: number;
        examine: string;
        [key: string]: any;  // For any additional properties
    } | number;  // For timestamps
}

export interface TimeseriesResponse {
    [itemId: string]: TimeseriesData[];
}

export class OSRSDataService {
    private readonly WIKI_API_BASE = 'https://api.weirdgloop.org/exchange/history/osrs';
    private readonly BULK_DATA_URL = 'https://chisel.weirdgloop.org/gazproj/gazbot/os_dump.json';
    private readonly USER_AGENT = 'OSRS-Market-Analysis-Tool/1.0';

    constructor() {
        // No need for setupHeaders anymore
    }

    async getItemHistory(itemId: string): Promise<TimeseriesData[]> {
        const response = await fetch(`${this.WIKI_API_BASE}/last90d?id=${itemId}`, {
            headers: {
                'User-Agent': this.USER_AGENT
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json() as TimeseriesResponse;
        return data[itemId] || [];
    }

    async getBulkData(): Promise<BulkDataResponse> {
        const response = await fetch(this.BULK_DATA_URL, {
            headers: {
                'User-Agent': this.USER_AGENT
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response.json() as Promise<BulkDataResponse>;
    }
}