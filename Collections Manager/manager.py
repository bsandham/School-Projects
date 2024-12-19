from flask import Flask, request, jsonify, render_template
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import regex as re
import requests
from datetime import datetime, timedelta
import math
import numpy as np
from numpy import griddata
from scipy.stats import norm
import matplotlib as plt

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
            
        underlying, yy, mm, dd, optType, strike = match.groups()
    
        basicData = {
            'underlying': underlying,
            'expiry': f'20{yy}-{mm}-{dd}',
            'type': 'call' if optType == 'C' else 'put',
            'strike': float(strike) / 1000,  # Convert from padded format
        }
        
        if df is not None:
            contractData = df[df['contractID'] == contractID]
            if not contractData.empty:
                basicData.update({
                    'bid': float(contractData['bid'].iloc[0]),
                    'ask': float(contractData['ask'].iloc[0]),
                    'volume': int(contractData['volume'].iloc[0]),
                    'impliedVol': float(contractData['impliedVol'].iloc[0])
                })
        
        return basicData
    
    @staticmethod
    def getField(contractID: str, field: str, df: Optional[pd.DataFrame] = None) -> Union[str, float, int]:
        """
        Get a specific field from parsed contract details
        """
        basicData = OptionParser.parseContract(contractID, df)
        if field not in basicData:
            raise ValueError(f"Field '{field}' not found. Available fields: {list(basicData.keys())}")
        return basicData[field]
    
    @staticmethod
    def getFields(contractID: str, fields: list, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        basicData = OptionParser.parseContract(contractID, df)
        return {field: basicData[field] for field in fields if field in basicData}

class DataLoader:
    def __init__(self):
        self.csvFile = 'contracts.csv'
        self.jsonFile = '/PATH'
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
            return OptionParser.getFields(contractID, fields, self.df)
        return OptionParser.parseContract(contractID, self.df)
    
    def getNewsfeed(self, underlying: str) -> list:
        apiKey = '' # prevent key being leaked for now pls replace
        daysBack = 7
        maxArticles = 1
        baseUrl = "https://api.webz.io/filterWebContent"
        
        endDate = datetime.now()
        startDate = endDate - timedelta(days=daysBack)
      
        params = {
            "token": apiKey,
            "q": f"\"{underlying}\" OR \"{underlying} stock\"",  
            "ts": f"{startDate.timestamp()}:{endDate.timestamp()}",
            "sort": "published",
            "size": maxArticles,
            "format": "json"
        }
        
        try:
            response = requests.get(baseUrl, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get('posts', []):
                articles.append({
                    'title': article.get('title'),
                    'publishedDate': article.get('published'),
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

class Calculations:
    def findGreeks(self, contractID: str) -> Dict[str, float]:
        greeksDict = {
            'delta': None,
            'gamma': None,
            'vega': None,
            'theta': None
        }
        
        contractData = OptionParser.getFields(contractID, ['underlyingPrice', 'type', 'strike', 'expiry', 'impliedVol'])
        expiry = datetime.strptime(contractData['expiry'], '%Y-%m-%d')
        today = datetime.now()

        daysToExpiry = (expiry-today).days
        yearsToExpiry = daysToExpiry/252

        sigma = float(contractData['impliedVol']/100) 
        S = float(contractData['underlyingPrice'])
        K = float(contractData['strike'])
        r = 4.5

        d1 = (math.log(S/K) + (r + ((sigma**2) * 0.5)*yearsToExpiry))/(sigma * math.sqrt(yearsToExpiry))
        d2 = d1 - sigma * np.sqrt(yearsToExpiry)
        
        if contractData['type'] == "call": 
            greeksDict['delta'] = d1
        else:
            greeksDict['delta'] = d1 - 1 
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(yearsToExpiry))
        greeksDict['gamma'] = gamma

        vega = S * np.sqrt(yearsToExpiry) * norm.pdf(d1)
        greeksDict['vega'] = vega/100

        if contractData['type'] == 'call':
            theta = (-(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(yearsToExpiry))) - r * K * np.exp(-r * yearsToExpiry) * norm.cdf(d2)
        else:
            theta = (-(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(yearsToExpiry))) + r * K * np.exp(-r * yearsToExpiry) * norm.cdf(-d2)

        greeksDict['theta'] = theta 
        return greeksDict

    def findPerformance(self, contractID: str, timescale: int) -> float:
        priceHistoryDF = pd.read_csv('priceHistory.csv')
        priceHistoryDF['priceDate'] = pd.to_datetime(priceHistoryDF['priceDate'], format='%y-%m-%d')

        contractData = priceHistoryDF[priceHistoryDF['contractID'] == contractID].copy()

        if contractData.empty:
            raise ValueError("No data for contract")
        
        contractData = contractData.sort_values('priceDate')

        endDate = contractData['priceDate'].max()
        startDate = endDate - timedelta(days=timescale)

        startPrice = contractData[contractData['priceDate'] >= startDate].iloc[0]['historicalPrice']
        endPrice = contractData[contractData['priceDate'] == endDate].iloc[0]['historicalPrice']

        percentChange = ((endPrice - startPrice) / startPrice) * 100
        return percentChange

    def findNotionalRisk(self, contractID: str) -> float:
        """ 
        notional risk = delta x underlyingPrice x 100 (1 option represents 100 of the underlying asset)
        """
        greeksDict = self.findGreeks(contractID)
        delta = greeksDict['delta']
        underlyingPrice = OptionParser.getField(contractID, 'underlyingPrice')
        notionalRisk = (delta * underlyingPrice * 100)
        return notionalRisk

class DFManipulation:
    def __init__(self): 
        self.filterTag = None   # BOOL TRUE/FALSE

    def filterDFByPrice(self, df: pd.DataFrame, filterTag: Optional[bool] = None) -> pd.DataFrame: 
        filterToUse = filterTag if filterTag is not None else self.filterTag
        return df.sort_values('price', ascending=filterToUse)

    def filterDFByIV(self, df: pd.DataFrame, filterTag: Optional[bool] = None) -> pd.DataFrame: 
        filterToUse = filterTag if filterTag is not None else self.filterTag
        return df.sort_values('impliedVol', ascending=filterToUse)
    
    def filterDFByName(self, df: pd.DataFrame, filterTag: Optional[bool] = None) -> pd.DataFrame: 
        filterToUse = filterTag if filterTag is not None else self.filterTag
        return df.sort_values('name', ascending=filterToUse)

class Visualizations:
    def modelVolSurface(self, name: str) -> None:
        allContracts = pd.read_csv('contracts.csv')
        allContractsFromUnderlying = allContracts[allContracts['name'] == name]

        def getSurfaceDataToNP(df: pd.DataFrame, fields: list) -> Dict[str, np.ndarray]:
            surfaceData = {
                'strikes': [],
                'expiries': [],
                'ivs': [],
                'underlyingPrice': None,
                'moneyness': None
            }

            for _, row in df.iterrows():
                contractData = OptionParser.getFields(row['contractID'], fields, df)

                surfaceData['strikes'].append(contractData.get('strike'))

                expiryDate = contractData.get('expiry')
                expiry = datetime.strptime(expiryDate, '%Y-%m-%d')
                today = datetime.now()
                dte = (expiry - today).days

                surfaceData['expiries'].append(dte/365)
                surfaceData['ivs'].append(contractData.get('impliedVol'))

                if surfaceData['underlyingPrice'] is None: 
                    surfaceData['underlyingPrice'] = contractData.get('underlyingPrice')

            surfaceData['strikes'] = np.array(surfaceData['strikes'])
            surfaceData['expiries'] = np.array(surfaceData['expiries'])
            surfaceData['ivs'] = np.array(surfaceData['ivs'])
            surfaceData['moneyness'] = surfaceData['strikes'] / surfaceData['underlyingPrice']

            return surfaceData
        
        fields = ['strike', 'expiry', 'impliedVol', 'underlyingPrice']
        surfaceData = getSurfaceDataToNP(allContractsFromUnderlying, fields)

        strikeRange = np.linspace(surfaceData['strikes'].min(), surfaceData['strikes'].max(), 50)
        expiryRange = np.linspace(surfaceData['expiries'].min(), surfaceData['expiries'].max(), 50)
        strikeMesh, expiryMesh = np.meshgrid(strikeRange, expiryRange)

        points = np.column_stack((surfaceData['strikes'], surfaceData['expiries']))
        ivMesh = griddata(points, surfaceData['ivs'], (strikeMesh, expiryMesh), method='cubic')

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        surface = ax.plot_surface(strikeMesh, expiryMesh, ivMesh, cmap='viridis', alpha=0.8)
        ax.scatter(surfaceData['strikes'], surfaceData['expiries'], surfaceData['ivs'], color='red', alpha=0.5)

        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Time to Expiry (Years)')
        ax.set_zlabel('Implied Volatility')
        plt.title(f'Implied Volatility Surface (Underlying Price: {surfaceData["underlyingPrice"]:.2f})')
        
        fig.colorbar(surface, ax=ax, label='Implied Volatility')
        plt.show()

    def modelPerformanceOverTime(self, contractID: str) -> pd.DataFrame:
        priceHistoryDF = pd.read_csv('priceHistory.csv')
        priceHistoryDF['priceDate'] = pd.to_datetime(priceHistoryDF['priceDate'], format='%y-%m-%d')

        contractData = priceHistoryDF[priceHistoryDF['contractID'] == contractID].copy()

        if contractData.empty:
            raise ValueError("No data for contract")
        
        contractData = contractData.sort_values('priceDate')
        contractData['MA5'] = contractData['historicalPrice'].rolling(window=5).mean()
        contractData['MA20'] = contractData['historicalPrice'].rolling(window=20).mean()
    
        plt.figure(figsize=(12, 6))
        
        plt.plot(contractData['priceDate'], 
                contractData['historicalPrice'],
                marker='o',
                linestyle='-',
                linewidth=2,
                label='Price',
                color='blue',
                alpha=0.6)
                
        plt.plot(contractData['priceDate'],
                contractData['MA5'],
                linestyle='-',
                linewidth=2,
                label='5-day MA',
                color='red')
                
        plt.plot(contractData['priceDate'],
                contractData['MA20'],
                linestyle='-',
                linewidth=2,
                label='20-day MA',
                color='green')

        plt.title(f'Price History and Moving Averages for Contract {contractID}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return contractData

class HelperFunctions(): 
    def __init__(self): 
        pass

    def exportToCSV(df): 
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