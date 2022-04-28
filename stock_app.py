import dash
import dash_daq as daq
#import dash_design_kit as ddk
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))

company=""
x_train,y_train=[],[]
model=None
prediction=None
train=valid=None
date=None

def sample(value):
    import yfinance as yf
    global company
    company = value
    #total=[]
    stock = yf.Ticker(company)
    df_nse = stock.history(period="10y")
    
    df_nse.insert(0, 'Date', df_nse.index)
    
    n=int(df_nse.shape[0]*0.8)
    d=60
    
    
    df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
    df_nse.index=df_nse['Date']
    
    
    data=df_nse.sort_index(ascending=True,axis=0)
    new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])
    
    for i in range(0,len(data)):
        new_data["Date"][i]=data['Date'][i]
        new_data["Close"][i]=data["Close"][i]
    
    new_data.index=new_data.Date
    new_data.drop("Date",axis=1,inplace=True)
    
    dataset=new_data.values
    
    global train
    global valid
    
    train=dataset[0:n,:]
    valid=dataset[n:,:]
    
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)
    
    
    global x_train
    global y_train
    
    x_train,y_train=[],[]
    for i in range(d,len(train)):
        x_train.append(scaled_data[i-d:i,0])
        y_train.append(scaled_data[i,0])
        
    x_train,y_train=np.array(x_train),np.array(y_train)
    
    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    
    global model
    model=load_model(company+".h5")
    
    inputs=new_data[len(new_data)-len(valid)-d:].values
    inputs=inputs.reshape(-1,1)
    inputs=scaler.transform(inputs)
    
    X_test=[]
    for i in range(d,inputs.shape[0]):
        X_test.append(inputs[i-d:i,0])
    X_test=np.array(X_test)
    
    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    closing_price=model.predict(X_test)
    closing_price=scaler.inverse_transform(closing_price)
    
    
    train=new_data[:n]
    valid=new_data[n:]
    valid['Predictions']=closing_price
    
    
    
    
    df1=df_nse[['Date','Close']]
    df1=df1.append(df1.iloc[-1:])
    
    dataset_train = df1.iloc[:n,1:2]
    dataset_test = df1.iloc[n:,1:2]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - d:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)
    
    my_test = [inputs[inputs.shape[0]-1-d : inputs.shape[0]-1, 0]]
    my_test = np.array(my_test)
    my_test = np.reshape(my_test, (my_test.shape[0], my_test.shape[1], 1))
    
    global prediction
    
    prediction = model.predict(my_test)
    prediction = scaler.inverse_transform(prediction)
    
    global date
    date = str(df1['Date'][-1]).split()
    date=date[0].split('-')
    date[2]=str(int(date[2])+1)
    date="-".join(date[::-1])

sample("WIPRO.NS")

df= pd.read_csv("./stock_data.csv") #markers

'''dcc.Input(
          id='text-input',
          type='text',
          value=company,
          placeholder='Enter ticker text for stock',
          ),'''
    

app.layout = html.Div([
   
    html.H1("Stock Market Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='Forecast Future Values of Stocks', children=[
            html.Div([
                html.H2("Description About Input Fields",style={"textAlign": "center", "text-decoration": "underline"}),
                html.P(
                    children=["1. Stock Name is the CAPITALIZED letters for the stock. AAPL fpr Apple, BTC-USD for Bitcoin, TSLA for Tesla.", html.Br(),html.Br(),
                        " 2. Epochs is the number of passes of the entire training dataset, the machine learning algorithm has completed. More the epochs, better the accuracy. Epoch can be set at 50 for a good performance.",html.Br(),html.Br(),
                        " 3. Days To Consider is the number of days you want to use as the prediction dataset to predict the future stock price. For example, you can use 30 days of data to predict the 31st day (ahead =1) or use the same data to predict 40th day (ahead=10).",html.Br(),html.Br(),
                        " 4. Days For Prediction is the number of days you want predict ahead of time. Less the number, higher the accuracy because model will have hard time predicting further into the future.",html.Br(),html.Br()],
                        style={"textAlign": "left", "color": "#000000", "font-size": "20px"}
                    ), 
                html.H2("Predict Future Stock Prices for Different Companies", style={"textAlign": "center", "text-decoration": "underline"}),
                html.Br(),
                html.Div([
                    html.H4("Stock Name", style={"textAlign": "center"}),
                    dcc.Dropdown(id='my-dropdown-list',
                             options=[{'label': 'Tata Steel Limited', 'value': 'TATASTEEL.NS'},
                                      {'label': 'Wipro Limited','value': 'WIPRO.NS'}, 
                                      {'label': 'Tata Consultancy Services Limited', 'value': 'TCS.NS'}, 
                                      {'label': 'Reliance Industries Limited','value': 'RELIANCE.NS'},
                                      {'label': 'Infosys Limited','value': 'INFY.NS'}], 
                             multi=False,value='',
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                    html.Div(id='company-basic', 
                             children='Select a company from dropdown above.', 
                             style={"padding" : "15px 32px", "textAlign" : "center", "font-size": "20px"}),
                    html.H4("Epochs", style={"textAlign": "center"}),
                    daq.NumericInput(
                        id='numeric-input1',
                        min=1,
                        max=100,
                        size = 100,
                        value=0
                    ),
                    html.Br(),
                    html.H4("Days To Consider", style={"textAlign": "center"}),
                    daq.NumericInput(
                        id='numeric-input2',
                        min=10,
                        max=100,
                        size = 100,
                        value=0
                    ),
                    html.Br(),
                    html.H4("Days for Prediction", style={"textAlign": "center"}),
                    daq.NumericInput(
                        id='numeric-input3',
                        min=1,
                        max=10,
                        size = 100,
                        value=0
                    ),
                    html.Br(),
                    html.Button('Submit', id='button', 
                                n_clicks=0, 
                                style={"padding" : "15px 32px","text-align" : "center", "display" : "inline-block", "font-size" : "16px"}),
                    html.Div(id='container-button-basic', 
                             children='Press submit to check next day stock price', 
                             style={"padding" : "15px 32px", "textAlign" : "center", "font-size": "20px"}),
                    html.Br(),
                    html.Br(),
                    html.Br()
                    
                ], style={"textAlign" : "center", "font-size": "20px"})
            ])
        ]),
        
        dcc.Tab(id="company-tab", label = 'Selected Stock Data',children=[
			html.Div([
				html.H2("Actual Closing Price",style={"textAlign" : "center"}),
				dcc.Graph(
					id="Actual Data",
                    
				),
				html.H2("LSTM Predicted Closing Price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data",
                    
				)				
			])        		


        ]),
        dcc.Tab(label='Visualize Stock Data', children=[
            html.Div([
                html.H2("Stocks High Values vs Low Values", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'}, 
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H2("Stocks Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])
    ])
])




'''@app.callback(
    [dash.dependencies.Input('text-input', 'value')])
def update_output(value):
    total.append(value)
'''


@app.callback(
    dash.dependencies.Output('company-basic', 'children'),
    [dash.dependencies.Input('my-dropdown-list', 'value')])
def update_output(value):
    if(value != ''):
        sample(value)
        return "You selected : ",value
    
@app.callback(
    dash.dependencies.Output('Actual Data', 'figure'),
    [dash.dependencies.Input('my-dropdown-list', 'value')])
def update_output(value):
    if(value != ''):
        figure={
			"data":[
					go.Scatter(
						x=train.index,
						y=train["Close"],
						mode='lines'
					)

				],
				"layout":go.Layout(
					title=value,
					xaxis={'title':'Date'},
					yaxis={'title':'Closing Rate'}
				)
		}
        return figure
@app.callback(
    dash.dependencies.Output('Predicted Data', 'figure'),
    [dash.dependencies.Input('my-dropdown-list', 'value')])
def update_output(value):
    if(value != ''):
        figure={
			"data":[
				{'x': valid.index, 'y': valid['Predictions'], 'type': 'scatter', 'name': 'Predicted Close'},
                {'x': valid.index, 'y': valid['Close'], 'type': 'scatter', 'name': 'Original Close'},

					],
			"layout":go.Layout(
				title=value,
				xaxis={'title':'Date'},
				yaxis={'title':'Closing Rate'}
				)
			}
        return figure

@app.callback(
    dash.dependencies.Output('container-button-basic', 'children'),
    [dash.dependencies.Input('button','n_clicks')])
def update_output(n_clicks):
    if(n_clicks > 0):
        return 'Next day Stock Price for {} ({}) is : {}'.format(company,date,prediction[-1])
    
    



@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure



if __name__=='__main__':
	app.run_server(debug=True)
