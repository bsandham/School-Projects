from flask import Flask, request, jsonify, render_template
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import regex as re
import requests
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self):
        self.csvFile = 'contracts.csv'
        self.JSONFile = '/PATH'
        self.df = None
        
    def getDataFromCSV(self):
        self.df = pd.read_csv(self.csvFile, 
                            names=['contract_id', 'description', 'underlying', 
                                  'strike', 'expiry', 'bid', 'ask', 'volume'])
        return self.df
        
    def getDataFromJSON(self):
        self.df = pd.read_json(self.JSONFile)
        return self.df
        
    def parseOptionDetails(self, contract_id):
        """
        Parse option contract details from contract ID
        Example: TSLA241213C00255000 -> 
        {
            'underlying': 'TSLA',
            'expiry': '2024-12-13',
            'type': 'call',
            'strike': 255.00
        }
        """
        if self.df is None:
            self.getDataFromCSV()
        
        contract_data = self.df[self.df['contract_id'] == contract_id]
        
        if contract_data.empty:
            return None
    
        # Format: UNDERLYING + YY + MM + DD + C/P + STRIKE(padded)
        pattern = r'([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d+)'
        match = re.match(pattern, contract_id)
        
        if not match:
            return None
            
        underlying, yy, mm, dd, opt_type, strike = match.groups()
        
        return {
            'underlying': underlying,
            'expiry': f'20{yy}-{mm}-{dd}',
            'type': 'call' if opt_type == 'C' else 'put',
            'strike': float(strike) / 1000,  
            'bid': float(contract_data['bid'].iloc[0]),
            'ask': float(contract_data['ask'].iloc[0]),
            'volume': int(contract_data['volume'].iloc[0])
        }
    
    def getNewsfeed(self, underlying):
        api_key = 'f6ec5702-a492-4964-b4cb-e66cfa1d773b'
        days_back = 7
        max_articles = 1
        base_url = "https://api.webz.io/filterWebContent"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
      
        params = {
            "token": api_key,
            "q": f"\"{underlying}\" OR \"{underlying} stock\"",  
            "ts": f"{start_date.timestamp()}:{end_date.timestamp()}",
            "sort": "published",
            "size": max_articles,
            "format": "json"
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get('posts', []):
                articles.append({
                    'title': article.get('title'),
                    'published_date': article.get('published'),
                    'source': article.get('thread', {}).get('site_full'),
                    'url': article.get('url'),
                    'text': article.get('text'),
                    'language': article.get('language')
                })
            
            return articles
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {str(e)}")
            return []
        
        except (KeyError, ValueError) as e:
            print(f"Error parsing response: {str(e)}")
            return []

         
class Calculations(): 
    def __init__(self): 
        pass 

    def findGreeks(self.contract): 
        pass 

    def findPerformance(self.contract, self.timescale): 
        pass 

    def findIV(self.contract): 
        pass 

    def findNotionalRisk(self.contract): 
        pass

class DFManipulation():
    def __init__(self): 
        pass

    def updateDF(self.dF):
        pass 

    def filterDF(self.dF, filter='ascending'): 
        pass 

class Viualisations():
    def __init__(self): 
        pass 

    def modelVolSurface(self.contract): 
        pass 

    def modelPerformanceOverTime(self.contract): 
        pass 

class HelperFunctions(): 
    def __init__(self): 
        pass 

    def exportToCSV(): 
        pass 

    def exportToJSON(): 
        pass 

class Application: 
    def __init__(self): 
        pass 

    def initialise(): 
        pass 

    def handle_request():
        pass 

app = Flask(__name__) 

@app.route('/')
def home(): 
    pass 

@app.route('/api/calculate/', methods=['POST'])
def calculate(): 
    pass 

@app.route('/api/data/', methods=['GET'])
def getContractData(contract: str):
    pass

@app.route('/api/news', methods=['POST'])
def getNewsUpdates(): 
    pass 

if __name__ == '__main__':
    instance = Application()
    instance.initialise(Path('data/input.csv'))
    app.run(debug=True)