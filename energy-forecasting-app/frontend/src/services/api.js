import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

class EnergyForecastingAPI {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  async getSampleData(hours = 168) {
    try {
      const response = await this.client.get(`/sample-data?hours=${hours}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching sample data:', error);
      throw error;
    }
  }

  async predictEnergyConsumption(data, modelType = 'both') {
    try {
      const response = await this.client.post('/predict', {
        data: data,
        model_type: modelType
      });
      return response.data;
    } catch (error) {
      console.error('Error making prediction:', error);
      throw error;
    }
  }

  async getModelInfo() {
    try {
      const response = await this.client.get('/model-info');
      return response.data;
    } catch (error) {
      console.error('Error fetching model info:', error);
      throw error;
    }
  }

  async healthCheck() {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      console.error('Error checking health:', error);
      throw error;
    }
  }
}

export default new EnergyForecastingAPI();
