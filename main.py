import pandas as pd
import math
import matplotlib.pyplot as plt
from numpy import *
from time import time
import numpy as np
import yfinance as yf
import scipy.stats as si
import sympy as sy
from sympy.stats import Normal, cdf
from sympy import init_printing
import cython
import math as m
from scipy.stats import norm, gmean
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/calcular")
async def calcular_opcao(cod_stock, preco_exercicio, tipo, p_c, inicio, risk_free, tempo):
    try:
        for param in [cod_stock, preco_exercicio, tipo, p_c, inicio, risk_free, tempo]:
            print(type(param))
        return calculadora(cod_stock, preco_exercicio, tipo, p_c, inicio, risk_free, tempo)

        print('teste')
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Item not found")
    return {"Hello": "World"}



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def coletapreco(ativo, datainicio):
    precos = yf.download(ativo, start = datainicio)
    precos = precos['Adj Close']
    S0=precos[-1]
    ret = precos.pct_change().dropna()
    sigma = np.std(ret) * np.sqrt(252)

    return S0, sigma


def euro_vanilla_call(S, K, T, r, sigma):
    # S: spot price
    # K: strike price
    # T: time to maturity
    # r: interest rate
    # sigma: volatility of underlying asset
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)


    for param in [S, K, T, r, sigma]:
        print(type(param))

    print('50% euro vanilla call')
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    print('50% euro vanilla call')

    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))

    return call


def euro_vanilla_put(S, K, T, r, sigma):
    # S: spot price
    # K: strike price
    # T: time to maturity
    # r: interest rate
    # sigma: volatility of underlying asset
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))

    return put


def asian_option_arithmetic_call(S0, K, T, r, sigma):
    K = float(K)
    T = float(T)
    r = float(r)
    for param in [S0, K, T, r, sigma]:
        print(type(param))

    M = 100
    I = 100000
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
    average_prices = np.mean(paths, axis=0)
    payoffs = np.maximum(average_prices - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def asian_option_arithmetic_put(S0, K, T, r, sigma):
    K = float(K)
    T = float(T)
    r = float(r)

    M = 100
    I = 100000
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
    average_prices = np.mean(paths, axis=0)
    payoffs = np.maximum(K - average_prices, 0)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price


def calculadora(ativo, preco_exercicio, tipo, p_c, inicio, risk_free, tempo):
    S0, sigma = coletapreco(ativo, inicio)
    print("S0, sigma")
    print(S0, sigma)
    print('pos coleta preco')
    if tipo == '1':

        if p_c == 'c':
            preco = euro_vanilla_call(S0, preco_exercicio, tempo, risk_free, sigma)
            print("O preço da sua opção é: " + str(preco))
        if p_c == 'p':
            preco = euro_vanilla_put(S0, preco_exercicio, tempo, risk_free, sigma)
            print("O preço da sua opção é: " + str(preco))

        return preco

    if tipo == '2':

        if p_c == 'c':
            preco = asian_option_arithmetic_call(S0, preco_exercicio, tempo, risk_free, sigma)
            print("O preço da sua opção é: " + str(preco))
        if p_c == 'p':
            preco = asian_option_arithmetic_put(S0, preco_exercicio, tempo, risk_free, sigma)
            print("O preço da sua opção é: " + str(preco))

        return preco

    return 'tipo de opcao inexistente'

if __name__ == '__main__':
    #uvicorn main:app --reload
    calculadora('VALE3.SA', 63, 1, 'c', '2020-02-02', 0.1065, 1)

