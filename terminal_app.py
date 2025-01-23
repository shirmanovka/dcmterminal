import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

def dcm_pricing_spread():
    st.header("Pricing spread")
    #st.title('Pricing spread')

    df = pd.read_excel('Карта рынка.xlsx', skiprows=1)
    df['spread'] = (df['Спред, пп'] * 100)
    df['Yield'] = ((100 - df['Цена, пп']) * 100) / df['Срок  до погашения / оферты, лет']
    df['Cupon'] = df['spread'] / df['Цена, пп'] * 100 - df['spread']
    df['Cspread'] = round(df['spread'] + df['Cupon'] + df['Yield'])
    
    
    s_df = df[['ISIN','Тикер','Рейтинг','Цена, пп', 'Срок  до погашения / оферты, лет','spread','Cspread']].copy()
    
    # Фильтры для столбцов
    tickers = s_df['Тикер'].unique()
    selected_tickers = st.multiselect('Выберите тикер:', tickers)
    
    # Фильтрация данных
    f_df = s_df[(s_df['Тикер'].isin(selected_tickers) | (len(selected_tickers) == 0))]
    
    # Отображение отфильтрованного DataFrame
    st.dataframe(f_df)
    
    if not f_df.empty:
        # Выбор облигации
        selected_bond = st.selectbox('Выберите облигацию:', f_df['Тикер'].unique())
        bond_data = f_df[f_df['Тикер'] == selected_bond].iloc[0]
    
        # Ввод цены или спреда
        input_type = st.radio("Выберите, что хотите ввести:", ("Цена", "Спред"))
    
        if input_type == "Цена":
            price_input = st.number_input("Введите цену облигации:", min_value=0.0, step=0.01)
            if st.button("Рассчитать спред"):
                # Расчет спреда на основе введенной цены
                spread_calculated = round(bond_data['spread'] + (bond_data['spread'] / price_input * 100 - bond_data['spread']) + (((100 - price_input) * 100) / bond_data['Срок  до погашения / оферты, лет']))
                st.success(f"Расчитанный спред: {spread_calculated:.2f}")
        
        elif input_type == "Спред":
            spread_input = st.number_input("Введите спред:", min_value=0.0, step=0.01)
            if st.button("Рассчитать цену"):
                # Расчет цены на основе введенного спреда
                
                price_calculated = ((100-spread_input*bond_data['Срок  до погашения / оферты, лет']/100)+((100-spread_input*bond_data['Срок  до погашения / оферты, лет']/100)**2+4*bond_data['Срок  до погашения / оферты, лет']*bond_data['spread'])**0.5)/2
    
                
                st.success(f"Расчитанная цена: {price_calculated:.2f}")



def dcm_matchbox():
    st.header("Matchbox")
    
    moex_url = 'https://iss.moex.com/iss/engines/stock/markets/bonds/boards/TQCB/securities.json'
    response = requests.get(moex_url) #получим ответ от сервера
    result = json.loads(response.text)
    col_name = result['securities']['columns'] # описываем какие колонки нахоядтся в матоданных #securuties или #history
    data_bonds_securities = pd.DataFrame(columns = col_name)
    # Часть_2 заполняем дата фрейм
    moex_url_securities = 'https://iss.moex.com/iss/engines/stock/markets/bonds/boards/TQCB/securities.json' #TQOB ОФЗ
    response = requests.get(moex_url_securities)
    result = json.loads(response.text)
    resp_date = result['securities']['data']
    securities_data_bonds = pd.DataFrame(resp_date, columns = col_name)
    a = len(resp_date)
    
    #Маленькая таблица сформированная из основной, показывающая краткий свод информации.
    securities_data_bonds = securities_data_bonds[securities_data_bonds['FACEUNIT'] == 'SUR']
    s_df = securities_data_bonds[['SECID',  'PREVLEGALCLOSEPRICE']].copy()
    s_df = s_df.rename(columns ={'SECID': 'ISIN'})
    s_df = s_df.rename(columns ={'PREVLEGALCLOSEPRICE': 'Цена, пп'})
    
     
    # Читаем файл xlsx
    df = pd.read_excel(('Карта рынка.xlsx'), skiprows=1)
    df = df.rename(columns ={'Цена, пп': 'Цена, пп1'}) # переименовал столбец чтобы его заменить
    
    df = pd.merge(s_df, df, on='ISIN', how='inner') #соеденил две таблицы, а дальше как в обычном расчете.
    
    df['Объем, млн'] = pd.to_numeric(df['Объем, млн'], errors='coerce')  # Преобразует в NaN некорректные значения
    # Формируем расчетные столбцы
    df['spread'] = (df['Спред, пп'] * 100)
    df['Yield'] = ((100 - df['Цена, пп']) * 100) / df['Срок  до погашения / оферты, лет']
    df['Cupon'] = df['spread'] / df['Цена, пп'] * 100 - df['spread']
    df['Cspread'] = round(df['spread'] + df['Cupon'] + df['Yield'])
    df['deltaS'] = round((df['Cspread'] - df['spread']),0)
    df['Name_rating_gap'] = df.apply(lambda row: f"{row['Тикер']}, {row['Рейтинг']}, {row['deltaS']}", axis=1)
    df['Размещениеt'] = pd.to_datetime(df['Размещение'], dayfirst=True)
    df['Размещениеt'] = df['Размещениеt'].dt.date
    df = df.sort_values(by='Размещениеt',ascending=True) #Cортируем от малых к большим
    
    #Создаем новый дата фрейм который и выводим на экран
    df1 = df[['ISIN', 'Тикер', 'Рейтинг', 'Валюта', 'Цена, пп', 
               'Срок  до погашения / оферты, лет', 'Частота купонных выплат', 
               'Базовая ставка', 'Опцион', 'Погашение','Размещениеt',
               'spread', 'Cspread', 'deltaS', 'Name_rating_gap']].copy()
    
    # Создаем Streamlit интерфейс
    # st.title('Matchbox')
    #st.write ('Description: уставновив филтры по рейтингу или дате размещения или выбрав в ручную можно построить карту по спредам флоуетров')
    st.write ('Данные отражены на Т-1')
    
              
    # Поле для ввода списка ISIN
    isin_input = st.text_area("Введите свои ISIN (по одному на строку):", height=150)
    
    # Преобразуем введенный текст в список ISIN
    input_isin_list = [line.strip() for line in isin_input.splitlines() if line.strip()]
     
    # Фильтры для столбцов
    tickers = df1['Тикер'].unique()
    selected_tickers = st.multiselect('Выберите тикер:', tickers)
     
    ratings = df1['Рейтинг'].unique()
    selected_ratings = st.multiselect('Выберите рейтинг:', ratings)
     
    # Фильтр по дате
    min_date = df1['Размещениеt'].min()
    max_date = df1['Размещениеt'].max()
    start_date = st.date_input('Выберите начальную дату:', min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input('Выберите конечную дату:', min_value=start_date, max_value=max_date, value=max_date)
    
    #Фильтрация данных
    f_df = df1[
        (df1['ISIN'].isin(input_isin_list) | (len(input_isin_list) == 0)) &
        (df1['Тикер'].isin(selected_tickers) | (len(selected_tickers) == 0)) &
        (df1['Рейтинг'].isin(selected_ratings) | (len(selected_ratings) == 0)) &
        (df1['Размещениеt'] >= start_date) &
        (df1['Размещениеt'] <= end_date)
    ]
    
    # Возможность удаления строк
    if not f_df.empty:
        # Позволяем пользователю выбрать строки для удаления по индексу
        indices_to_delete = st.multiselect(
            'Выберите строки для удаления (по индексу):', options=f_df.index.tolist(), default=[]
        )
        
        # Удаляем выбранные строки из отфильтрованного DataFrame
        f_df = f_df.drop(index=indices_to_delete)
    
    # Отображение отфильтрованного DataFrame
    st.dataframe(f_df)
     
    # Построение графика
    if not f_df.empty:
        plt.figure(figsize=(12, 6))
     
        plt.scatter(f_df['Размещениеt'], f_df['Cspread'], color='darkred', marker='o', s=80, label='Текущий спред')
        plt.scatter(f_df['Размещениеt'], f_df['spread'], color='tan', marker='o', s=80, label='Спред при размещении')
     
        for i, row in f_df.iterrows():
            plt.text(row['Размещениеt'], row['spread'] + 4, row['Name_rating_gap'], ha='left', fontsize=10)
            
        for i in range(len(f_df)):
            for j in range(len(f_df)):
                        if f_df['Размещениеt'].iloc[i] == f_df['Размещениеt'].iloc[j]:
                        
                                plt.annotate ('', xy = (f_df['Размещениеt'].iloc[j], f_df['Cspread'].iloc[j]),
                                                xytext=(f_df['Размещениеt'].iloc[i], f_df['spread'].iloc[i]),
                                                arrowprops =dict(arrowstyle='->', color='goldenrod', linewidth=2, shrinkA=7,shrinkB=7)) #Рисуем стрелки над точками.    
     
        plt.title('Карта рынка', fontsize=18)
        plt.xlabel('Дата размещения', fontsize=16)
        plt.ylabel('Спред к КС', fontsize=16)
        plt.legend()
        plt.grid()
        plt.xticks(rotation=45)
     
        # Показываем график в Streamlit
        st.pyplot(plt)
    else:
        st.write("Нет данных для отображения.")
    
    
    
    st.write("*Если облигационный выпуск имеет амортизацию, то расчет изменения спреда ее не учитывает.")


def dcm_Mat_val():
    st.header("Maturity volume")
    # Вставьте код из приложения DCM Terminal
    #st.write ('Description: по облигациям с фиксированным купоном и рейтингом не ниже А- выбрав период погшения можно построить график погашения')
# Заголовок приложения
    #st.title("Фильтр данных по погашению")

    df = pd.read_excel('Карта рынка.xlsx', skiprows=1)
    df_1 = pd.read_excel('Карта рынка fix.xlsx', skiprows=1)
    
    df_1['Базовая ставка'] ='fix'
    df['Базовая ставка'] = 'floater'
    
    
    df['Погашение'] = pd.to_datetime(df['Погашение'], format='%d-%m-%Y', errors='coerce')
    df_1['Погашение'] = pd.to_datetime(df_1['Погашение'], format='%d-%m-%Y', errors='coerce')
    
    # Очистка и преобразование столбца 'Объем, млн'
    df['Объем, млн'] = df['Объем, млн'].astype(str).str.replace("'", "", regex=False)
    df['Объем, млн'] = pd.to_numeric(df['Объем, млн'], errors='coerce')
    
    df_1['Объем, млн'] = df_1['Объем, млн'].astype(str).str.replace("'", "", regex=False)
    df_1['Объем, млн'] = pd.to_numeric(df_1['Объем, млн'], errors='coerce')
    
    #Создание малых дата фреймов
    df1 = df[['ISIN', 'Тикер', 'Рейтинг', 'Валюта', 'Объем, млн', 
             'Погашение','Опцион', 'Базовая ставка' ]].copy()
    
    df_2 = df_1[['ISIN', 'Тикер', 'Рейтинг', 'Валюта', 'Объем, млн', 
             'Погашение','Опцион','Базовая ставка' ]].copy()
    
    
    # Заполнить отсутствующие столбцы значением np.nan
    df1 = df1.reindex(columns=df_2.columns)
    df_2 = df_2.reindex(columns=df1.columns)
    
    # Объединить df1 и df_2
    df3 = pd.concat([df1, df_2], ignore_index=True)
    
    
    # Проверка, какие даты доступны для фильтрации
    min_date = df3['Погашение'].min()
    max_date = df3['Погашение'].max()
    
    # Выбор диапазона дат для фильтрации
    start_date = st.date_input("Выберите начальную дату", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("Выберите конечную дату", min_value=min_date, max_value=max_date, value=max_date)
    
    # Поле для ввода списка ISIN
    isin_input = st.text_area("Введите свои ISIN (по одному на строку):", height=150)
    
    # Преобразуем введенный текст в список ISIN
    input_isin_list = [line.strip() for line in isin_input.splitlines() if line.strip()]
    
    # Фильтры для столбцов Тикер и Рейтинг
    tickers = df3['Тикер'].unique()
    selected_tickers = st.multiselect('Выберите тикер:', tickers)
    
    # Фильтрация по валюте
    unique_currencies = df3['Валюта'].unique()  # Получаем уникальные валюты
    selected_currency = st.multiselect("Выберите валюту", unique_currencies)  # Выбор валюты
    
    
    # Фильтрация данных
    filtered_df = df3[
        (df3['ISIN'].isin(input_isin_list) | (len(input_isin_list) == 0)) &
        (df3['Тикер'].isin(selected_tickers) | (len(selected_tickers) == 0)) &
        (df3['Погашение'] >= pd.Timestamp(start_date)) &
        (df3['Погашение'] <= pd.Timestamp(end_date)) |
        (df3['Валюта'].isin(selected_currency))
    ]
    
    
    # Вывод отфильтрованных данных
    st.write("Отфильтрованные данные:")
    st.dataframe(filtered_df)
    
    # Визуализация данных (если есть отфильтрованные данные)
    if not filtered_df.empty:
        # Создание графика с использованием Plotly
        fig = go.Figure(data=[
            go.Bar(
                x=filtered_df['Погашение'],
                y=filtered_df['Объем, млн'],
                text=filtered_df['Тикер'],  # Подписи для всплывающих подсказок
                hoverinfo='text',  # Показать только текст при наведении
                marker_color='darkred'
            )
        ])
    
        # Обновление макета графика
        fig.update_layout(
            title={
                'text': 'График погашений',
                'font': {'size': 18},  # Размер шрифта заголовка
                'x': 0.5,  # Центрирование заголовка по оси X
                'xanchor': 'center'  # Якорь заголовка
            },
            xaxis_title='Дата погашения',
            yaxis_title='Объем, млн',
            xaxis_tickformat='%Y-%m-%d',
            xaxis_title_font={'size': 16},  # Размер шрифта подписи по оси X
            yaxis_title_font={'size': 16} 
        )
    
        # Отображение графика в Streamlit
        st.plotly_chart(fig)
    
    else:
        st.write("Нет данных для отображения с выбранными параметрами.")
    st.write("*Дата погашения не учитвает оферту.")
    

def dcm_Currency_Indexes_Swap():
    
    st.header("Currency, Indexes, Swap Curve")
    def get_data(url):
        response = requests.get(url)
        result = response.json()
        col_names = result['marketdata']['columns']
        data = pd.DataFrame(result['marketdata']['data'], columns=col_names)
        return data
    
    def load_rgbi():
        moex_url = 'https://iss.moex.com/iss/engines/stock/markets/index/securities/RGBI.json'
        df = get_data(moex_url)
        return df
    
    def load_imoex():
        moex_url = 'https://iss.moex.com/iss/engines/stock/markets/index/securities/IMOEX.json'
        df = get_data(moex_url)
        return df

    def load_RUFLCBCP():
        moex_url = 'https://iss.moex.com/iss/engines/stock/markets/index/securities/RUFLCBCP.json'
        df = get_data(moex_url)
        return df
    
    def load_RUFLBICP():
        moex_url = 'https://iss.moex.com/iss/engines/stock/markets/index/securities/RUFLBICP.json'
        df = get_data(moex_url)
        return df
    
    # Функция для получения данных ставки ЦБ РФ
    def get_exchange_rates():
        moex_url_cbrf = 'https://iss.moex.com//iss/statistics/engines/currency/markets/selt/rates.json'
        
        try:
            response = requests.get(moex_url_cbrf)
            if response.status_code == 200:
                result = response.json()
                col_names = result['cbrf']['columns']
                df = pd.DataFrame(result['cbrf']['data'], columns=col_names)
                
                selected_columns = [
                    'CBRF_USD_LAST',
                    'CBRF_USD_LASTCHANGEPRCNT',
                    'CBRF_USD_TRADEDATE',
                    'CBRF_EUR_LAST',
                    'CBRF_EUR_LASTCHANGEPRCNT',
                    'CBRF_EUR_TRADEDATE'
                ]
                filtered_df = df[selected_columns]
                return filtered_df
            else:
                st.error(f'Ошибка при получении данных. Код состояния: {response.status_code}')
        except Exception as e:
            st.error(f'Произошла ошибка при запросе данных: {e}')
    def get_swap_curves():
        moex_url = 'https://iss.moex.com//iss/sdfi/curves/securities.json'
        
        try:
            response = requests.get(moex_url)
            if response.status_code == 200:
                result = response.json()
                col_names = result['curves']['columns']
                df = pd.DataFrame(result['curves']['data'], columns=col_names)
                return df
            else:
                st.error(f'Ошибка при получении данных. Код состояния: {response.status_code}')
        except Exception as e:
            st.error(f'Произошла ошибка при запросе данных: {e}')
    
    
    # Блок с данными ставки ЦБ РФ
    st.header("Курс рубля, ЦБ РФ")
    
    # Получаем данные о курсах валют
    exchange_rates = get_exchange_rates()
    
    if exchange_rates is not None:
        usd_last = exchange_rates['CBRF_USD_LAST'].values[0]
        usd_change = float(exchange_rates['CBRF_USD_LASTCHANGEPRCNT'][0])
        usd_trade_date = pd.to_datetime(exchange_rates['CBRF_USD_TRADEDATE']).dt.date.values[0]
        
        eur_last = exchange_rates['CBRF_EUR_LAST'].values[0]
        eur_change = float(exchange_rates['CBRF_EUR_LASTCHANGEPRCNT'].values[0])
        eur_trade_date = pd.to_datetime(exchange_rates['CBRF_EUR_TRADEDATE']).dt.date.values[0]
        
        # Размещаем курсы валют в колонках
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"USD: {usd_last}")
            change_color = "green" if usd_change >= 0 else "red"
            st.markdown(f"Изменение к закрытию: <span style='color:{change_color}; font-weight:bold; font-size:16px;'>{usd_change:.2f}%</span>", unsafe_allow_html=True)
            st.text(f"Дата обновления: {usd_trade_date}")
        
        with col2:
            st.subheader(f"EUR: {eur_last}")
            change_color = "green" if eur_change >= 0 else "red"
            st.markdown(f"Изменение к закрытию: <span style='color:{change_color}; font-weight:bold; font-size:16px;'>{eur_change:.2f}%</span>", unsafe_allow_html=True)
            st.text(f"Дата обновления: {eur_trade_date}")
    
    # Индексы RGBI и IMOEX
    st.header("Индексы")
    left_column, right_column = st.columns(2)
    
    with left_column:
        st.subheader(f"RGBI: {load_rgbi()['CURRENTVALUE'].values[0]}")
        
        rgbi_df = load_rgbi()
        
        last_change = float(rgbi_df['LASTCHANGEPRC'].values[0])
        change_color = "green" if last_change >= 0 else "red"
        st.markdown(f"Изменение к закрытию: <span style='color:{change_color}; font-weight:bold; font-size:16px;'>{last_change:.2f}%</span>", unsafe_allow_html=True)
        st.text(f"Дата обновления: {rgbi_df['SYSTIME'].values[0]}")
    
    with right_column:
        st.subheader(f"IMOEX: {load_imoex()['CURRENTVALUE'].values[0]}")
        
        imoex_df = load_imoex()
        
        last_change = float(imoex_df['LASTCHANGEPRC'].values[0])
        change_color = "green" if last_change >= 0 else "red"
        st.markdown(f"Изменение к закрытию: <span style='color:{change_color}; font-weight:bold; font-size:16px;'>{last_change:.2f}%</span>", unsafe_allow_html=True)
        st.text(f"Дата обновления: {imoex_df['SYSTIME'].values[0]}")
# Индексы RUFLBICP и RUFLCBCP ценовые

    left_column, right_column = st.columns(2)
    
    with left_column:
        st.subheader(f"Corp RUFLCBCP: {load_RUFLCBCP()['CURRENTVALUE'].values[0]}")
        
        RUFLCBCP_df = load_RUFLCBCP()
        
        last_change = float(RUFLCBCP_df['LASTCHANGEPRC'].values[0])
        change_color = "green" if last_change >= 0 else "red"
        st.markdown(f"Изменение к закрытию: <span style='color:{change_color}; font-weight:bold; font-size:16px;'>{last_change:.2f}%</span>", unsafe_allow_html=True)
        st.text(f"Дата обновления: {RUFLCBCP_df['SYSTIME'].values[0]}")
    
    with right_column:
        st.subheader(f"Float RUFLBICP: {load_RUFLBICP()['CURRENTVALUE'].values[0]}")
        
        RUFLBICP_df = load_RUFLBICP()
        
        last_change = float(RUFLBICP_df['LASTCHANGEPRC'].values[0])
        change_color = "green" if last_change >= 0 else "red"
        st.markdown(f"Изменение к закрытию: <span style='color:{change_color}; font-weight:bold; font-size:16px;'>{last_change:.2f}%</span>", unsafe_allow_html=True)
        st.text(f"Дата обновления: {RUFLBICP_df['SYSTIME'].values[0]}")
    

    
    # Блок с графиками кривых свопов
    st.header("Графики кривых свопов")
    
    # Автоматический запрос данных
    curves_data = get_swap_curves()
    
    if curves_data is not None:
    # Убедимся, что столбец 'swap_curve' существует
        if 'swap_curve' in curves_data.columns:
            swap_curve_filter = st.selectbox('Выберите кривую:', options=curves_data['swap_curve'].unique())
            filtered_data = curves_data.query(f"swap_curve == '{swap_curve_filter}'")
            
            # Получение даты выгрузки
            trade_date_str = filtered_data['tradedate'].values[0]
            trade_date = datetime.strptime(trade_date_str, '%Y-%m-%d').strftime('%d.%m.%Y')  # Преобразуем формат даты
            
            # Выводим дату выгрузки
            st.write(f"Дата выгрузки: {trade_date}")
            
            # Строим график
            fig = px.line(filtered_data,
                          x='tenor',
                          y='swap_rate',
                          title=f'Кривая свопа "{swap_curve_filter}"',
                          color_discrete_sequence=['darkred']  # Устанавливаем цвет линии на красный
                         )
            
            # Настройка подписей осей
            fig.update_layout(xaxis_title='Период',
                              yaxis_title='Ставка',
                             )
            
            st.plotly_chart(fig, use_container_width=True)  
            
        if st.button('Обновить данные', key='refresh'):
            st.script_runner.rerun()

def dcm_Currency_Indexes_Swap():
    
    st.header("Listing application")
  
    
    # Заголовки для запроса
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    # Функция для скачивания файла и чтения данных
    def download_data():
        url = 'https://web.moex.com/moex-web-icdb-api/api/v1/export/listing-applications/xlsx'
        response = requests.get(url, headers=headers)
    
        if response.status_code == 200:
            with open('listing_applications.xlsx', 'wb') as f:
                f.write(response.content)
            return pd.read_excel('listing_applications.xlsx')
        else:
            st.error("Ошибка при скачивании данных.")
            return None
    
    # Заголовок приложения
    st.title("Скачивание данных о заявках на листинг")
    
    # Кнопка для скачивания данных
    if st.button("Скачать данные"):
        df = download_data()
        if df is not None:
            columns_to_keep = [
            'Наименование эмитента',
            'ИНН эмитента',
            'Категория(тип) ценной бумаги',
            'Идентификатор выпуска*',
            'Уровень листинга',
            'Дата получения заявления',
            'Дата раскрытия информации'
        ]
            df = df[columns_to_keep]
            df['Категория(тип) ценной бумаги'] = df['Категория(тип) ценной бумаги'].str.replace(' ', '\n')
            
            abbreviation = {
           'Общество с ограниченной ответственностью': 'OOO',
           'Публичное акционерное общество': 'ПAO',
           'Акционерное общество': 'AO'
            }
    
       # Замена полных названий на аббревиатуры
            df['Наименование эмитента'] = df['Наименование эмитента'].replace(abbreviation)
            
            def highlight_rows(row):
                return ['background-color: lightgreen' if row['Идентификатор выпуска*'] == "Не присвоен" else '' for _ in row]
    
            new_df = df[df['Идентификатор выпуска*'] == "Не присвоен"]
            new_df = new_df[['Наименование эмитента', 'Категория(тип) ценной бумаги', 'Идентификатор выпуска*']].tail(5)
    
            styled_new_df = new_df.style.apply(highlight_rows, axis=1)  # Применение стиля к новому DataFrame
            
            df = df.tail(10)
            
            styled_df = df.style.apply(highlight_rows, axis=1)
    
    
            for full_name, abbreviation in abbreviation.items():
                df['Наименование эмитента'] = df['Наименование эмитента'].replace(full_name, abbreviation, regex=True)
            
            styled_df = df.style.apply(highlight_rows, axis=1)    
                # Вывод последних 5 строк DataFrame с примененной стилизацией
                # Новый DataFrame для условия "Не присвоен"
    
            
            st.markdown("""
                <style>
                    .dataframe th, .dataframe td {
                        padding: 5px;
                        text-align: left;
                        font-size: 12px;
                        white-space: pre-wrap; /* Позволяет переносить текст */
                    }
                </style>
            """, unsafe_allow_html=True)
    
            
            st.dataframe(styled_df.format({'ИНН эмитента': '{:.0f}'}), use_container_width=True)  # Основной DataFrame
            
            st.subheader("Последние заявки эмитентов на регистрацию нового выпуска")
            st.dataframe(styled_new_df, use_container_width=True)  # Новый DataFrame


def main():
    st.sidebar.title("DCM analytical terminal")
    menu = st.sidebar.selectbox("Выберите приложение", ("Pricing spread", "Matchbox", "Maturity volume","Currency, Indexes, Swap Curve", "Listing application"))

    if menu == "Pricing spread":
        dcm_pricing_spread()
    elif menu == "Matchbox":
        dcm_matchbox()
    elif menu == "Maturity volume":
        dcm_Mat_val()
    elif menu == "Currency, Indexes, Swap Curve":
        dcm_Currency_Indexes_Swap()
    elif menu == "Listing application":
        dcm_Listing_application()    

if __name__ == "__main__":
    main()
