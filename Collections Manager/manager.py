from flask import Flask, request, jsonify, render_template
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import regex as re
import requests
from datetime import datetime, timedelta
import math

class OptionParser:
    """utility class to get the specifics from a given contract id"""
    
    @staticmethod
    def parseContract(contractID: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        find all specific fields re: any given contract then optionally merge with DataFrame data
        """
        pattern = r'([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d+)'
        match = re.match(pattern, contractID)
        
        if not match:
            raise ValueError(f"Contract ID Invalid. {contractID}")
            
        underlying, yy, mm, dd, opt_type, strike = match.groups()
    
        basicData = {
            'underlying': underlying,
            'expiry': f'20{yy}-{mm}-{dd}',
            'type': 'call' if opt_type == 'C' else 'put',
            'strike': float(strike) / 1000,  # Convert from padded format
        }
        
        if df is not None:
            contract_data = df[df['contractID'] == contractID]
            if not contract_data.empty:
                basicData.update({
                    'bid': float(contract_data['bid'].iloc[0]),
                    'ask': float(contract_data['ask'].iloc[0]),
                    'volume': int(contract_data['volume'].iloc[0]),
                    'impliedVol': int(contract_data['impliedVol'].iloc[0])
                })
        
        return basicData
    
    @staticmethod
    def getField(contractID: str, field: str, df: Optional[pd.DataFrame] = None) -> Union[str, float, int]:
        """
        Get a specific field from parsed contract details
        
        Args:
            contractID: The option contract ID
            field: Field to retrieve ('underlying', 'expiry', 'type', 'strike', 'bid', 'ask', 'volume', 'impliedVol')
            df: Optional DataFrame with market data
        
        Returns:
            The requested field value
        """
        basicData = OptionParser.parse_contract(contractID, df)
        if field not in basicData:
            raise ValueError(f"Field '{field}' not found. Available fields: {list(basicData.keys())}")
        return basicData[field]
    
    @staticmethod
    def getFields(contractID: str, fields: list, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        basicData = OptionParser.parse_contract(contractID, df)
        return {field: basicData[field] for field in fields if field in basicData}

class DataLoader:
    def __init__(self):
        self.csvFile = 'contracts.csv'
        self.JSONFile = '/PATH'
        self.df = None
        
    def getDataFromCSV(self):
        self.df = pd.read_csv(self.csvFile, 
                            names=['contractID', 'description', 'underlying', 
                                  'strike', 'expiry', 'bid', 'ask', 'volume', 'impliedVol'])
        return self.df
    
    def parseOptionDetails(self, contractID: str, fields: Optional[list] = None):
        """
        Wrapper method to use OptionParser with the loaded DataFrame
        """
        if self.df is None:
            self.getDataFromCSV()
            
        if fields:
            return OptionParser.get_fields(contractID, fields, self.df)
        return OptionParser.parse_contract(contractID, self.df)
    
    def getNewsfeed(self, underlying):
        api_key = '' # prevent key being leaked for now pls replace
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

    def findGreeks(self.contractID): # these are european options, so we need to use the black/scholes formula
        contractData = OptionParser.getFields(self.contractID, ['underlyingPrice', 'type', 'strike', 'expiration', 'impliedVol'])
        expiry = datetime.strptime(contractData['expiration'], '%Y-%m-%d')
        today = datetime.now()

        daysToExpiry = (expiry-today).days
        yearsToExpiry = daysToExpiry/252

        sigma = float(contractData['implied_vol']/100) 
        S = float(contractData['underlyingPrice'])
        K = float(contractData['strike'])
        r = 4.5

        d1 = (math.log(S/K) + (r + ((sigma**2)/2)*yearsToExpiry))/(sigma * math.sqrt(yearsToExpiry))  # delta calculation
        
        if contractData['type'] == "call": 
            return d1
        else:
            return (d1 - 1)



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