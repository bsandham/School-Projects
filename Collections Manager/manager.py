from flask import Flask, request, jsonify, render_template
import pandas as pd
import yfinance as yf
from typing import Dict, List, Any, Optional
from pathlib import Path

class DataLoader(): 
    def __init__(self):
        self.csvFile = 'contracts.csv'
        self.JSONFile = '/PATH'

    def getDataFromCSV(self.CSVFile): 
        pass 

    def getDataFromJSON(self.JSONFile): 
        pass 

    def getNewsfeed():
        pass 

class ProcessData():
    def __init__(self): 
        pass 

    def saveToDF():
        pass 

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