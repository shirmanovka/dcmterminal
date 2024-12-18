import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
import json
import plotly.graph_objects as go

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
    st.write ('Description: уставновив филтры по рейтингу или дате размещения или выбрав в ручную можно построить карту по спредам флоуетров')
    st.write ('Данные отражены наТ-1')
    
              
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
   st.write ('Description: по облигациям с фиксированным купоном и рейтингом не ниже А- выбрав период погшения можно построить график погашения')
# Заголовок приложения
    #st.title("Фильтр данных по погашению")
    
    # Чтение данных из Excel
    df = pd.read_excel('Карта рынка fix.xlsx', skiprows=1)
    
    # Преобразование колонки 'Погашение' в формат datetime
    df['Погашение'] = pd.to_datetime(df['Погашение'], format='%d-%m-%Y', errors='coerce')
    
    # Очистка и преобразование столбца 'Объем, млн'
    df['Объем, млн'] = df['Объем, млн'].astype(str).str.replace("'", "", regex=False)
    df['Объем, млн'] = pd.to_numeric(df['Объем, млн'], errors='coerce')
    
    # Проверка, какие даты доступны для фильтрации
    min_date = df['Погашение'].min()
    max_date = df['Погашение'].max()
    
    # Выбор диапазона дат для фильтрации
    start_date = st.date_input("Выберите начальную дату", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("Выберите конечную дату", min_value=min_date, max_value=max_date, value=max_date)
    
    # Фильтрация по валюте
    unique_currencies = df['Валюта'].unique()  # Получаем уникальные валюты
    selected_currency = st.multiselect("Выберите валюту", unique_currencies)  # Выбор валюты
    
    # Фильтрация по диапазону дат
    filtered_df = df[(df['Погашение'] >= pd.Timestamp(start_date)) & 
                      (df['Погашение'] <= pd.Timestamp(end_date)) & 
                      (df['Валюта'].isin(selected_currency))]
    
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
            title='График погашений',
            xaxis_title='Дата погашения',
            yaxis_title='Объем, млн',
            xaxis_tickformat='%Y-%m-%d'
        )
    
        # Отображение графика в Streamlit
        st.plotly_chart(fig)
    else:
        st.write("Нет данных для отображения с выбранными параметрами.")




def main():
    st.sidebar.title("DCM analytical terminal")
    menu = st.sidebar.selectbox("Выберите приложение", ("Pricing spread", "Matchbox", "Maturity volume"))

    if menu == "Pricing spread":
        dcm_pricing_spread()
    elif menu == "Matchbox":
        dcm_matchbox()
    elif menu == "Maturity volume":
        dcm_Mat_val()

if __name__ == "__main__":
    main()
