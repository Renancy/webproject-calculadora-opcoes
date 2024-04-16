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

def opcao_compra_americana(S, K, T, r, sigma):
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)


    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    preco = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return preco

def opcao_venda_americana(S, K, T, r, sigma):
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)


    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    preco = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return preco

def volatilidade_put(S, K, T, r, vol):
    """
    S: preço atual
    K: preço de exercício
    T: tempo até a maturidade
    r: taxa de juros
    vol: volatilidade do ativo subjacente
    """
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    vol = float(vol)


    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))

    return put


def volatilidade_call(S, K, T, r, vol):
    """
    S: preço atual
    K: preço de exercício
    T: tempo até a maturidade
    r: taxa de juros
    vol: volatilidade do ativo subjacente
    """
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    vol = float(vol)

    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))

    return call

def cliquet_put(S, K, T, r, sigma, periods):
    """
    S: preço atual
    K: preço de exercício
    T: tempo até a maturidade
    r: taxa de juros
    sigma: volatilidade do ativo subjacente
    periods: número de períodos de cliquet
    """
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)
    periods = float(periods)


    dt = T / periods
    n = periods
    p = 0  # acumulador de cliquet

    for i in range(1, n+1):
        t = i * dt
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        p += (K * np.exp(-r * t) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))

    put = p / n

    return put



def cliquet_call(S, K, T, r, sigma, periods):
    """
    S: preço atual
    K: preço de exercício
    T: tempo até a maturidade
    r: taxa de juros
    sigma: volatilidade do ativo subjacente
    periods: número de períodos de cliquet
    """
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)
    periods = float(periods)


    dt = T / periods
    n = periods
    c = 0  # acumulador de cliquet

    for i in range(1, n+1):
        t = i * dt
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        c += (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * t) * si.norm.cdf(d2, 0.0, 1.0))

    call = c / n

    return call



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
    if tipo == '3':
        if p_c == 'c':
            preco = opcao_compra_americana(S0, preco_exercicio, tempo, risk_free, sigma)
            print("O preço da sua opção americana de compra é: " + str(preco))
        elif p_c == 'p':
            preco = opcao_venda_americana(S0, preco_exercicio, tempo, risk_free, sigma)
            print("O preço da sua opção americana de venda é: " + str(preco))

        return preco
    if tipo == '4':

        if p_c == 'c':
            preco = cliquet_call(S0, preco_exercicio, tempo, risk_free, sigma, periods=100)
            print("O preço da sua opção cliquet é: " + str(preco))
        if p_c == 'p':
            preco = cliquet_put(S0, preco_exercicio, tempo, risk_free, sigma, periods=100)
            print("O preço da sua opção cliquet é: " + str(preco))
        return preco

    if tipo == '5':

        if p_c == 'c':
            preco = volatilidade_call(S0, preco_exercicio, tempo, risk_free, sigma)
            print("O preço da sua opção de volatilidade é: " + str(preco))
        if p_c == 'p':
            preco = volatilidade_put(S0, preco_exercicio, tempo, risk_free, sigma)
            print("O preço da sua opção de volatilidade é: " + str(preco))
        return preco
    return 'tipo de opcao inexistente'

if __name__ == '__main__':
    #uvicorn main:app --reload
    calculadora('VALE3.SA', 63, 1, 'c', '2020-02-02', 0.1065, 1)

