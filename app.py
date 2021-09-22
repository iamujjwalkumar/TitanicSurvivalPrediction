from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

UPLOADED = False
SPLITTED = False
TRAINEDMODEL = False
PREPROCESSED = False


def PreprocessData(dataFrame):
    global PREPROCESSED
    if PREPROCESSED==False:
        PREPROCESSED=True
        for i in list(dataFrame.columns):
            if (dataFrame[i].isna().sum() > 0):
                data = {}
                df = dataFrame[i].fillna("null")
                for j in df:
                    if j != "null":
                        if j in data.keys():
                            data[j] += 1
                        else:
                            data[j] = 1
                temp = 0
                for j, k in data.items():
                    if temp < k:
                        maxOcc = j
                dataFrame[i] = dataFrame[i].fillna(maxOcc)

            data = list(dataFrame[i].value_counts().index)
            df = dataFrame[i]
            for j in df:
                df = df.replace(j, data.index(j))
            dataFrame[i] = df
    return dataFrame

@app.route('/')
def index():  # put application's code here
    return render_template('index.html')

@app.route('/UploadFile')
def UploadFile():  # put application's code here
    return render_template('UploadFile.html')

@app.route('/fetchdata',methods=['POST','GET'])
def fetchdata():
    global UPLOADED,SPLITTED,TRAINEDMODEL
    if request.method == 'POST':
        fileName = request.files['fileName']
        if fileName.filename=="":
            return render_template('UploadFile.html', UPLOADED=UPLOADED)
        fileName.save("DataSets/"+secure_filename(fileName.filename))#to get filename: "obj.filename"
        global dataFrame
        dataFrame = pd.read_csv("DataSets/"+fileName.filename)
        UPLOADED = True
        SPLITTED = False
        TRAINEDMODEL = False
        dataFrame = PreprocessData(dataFrame)
    return render_template('UploadFile.html',UPLOADED=UPLOADED,data=dataFrame)

@app.route('/removeAttributes',methods=['POST','GET'])
def removeAttributes():
    global UPLOADED,dataFrame
    if request.method == 'POST':
        fileName = request.form.get('Attributes').split(",")
        if fileName!=[]:
            dataFrame = dataFrame.drop(fileName, axis='columns')
    return render_template('UploadFile.html',UPLOADED=UPLOADED,data=dataFrame)


@app.route('/ViewData')
def ViewData():  # put application's code here
    global UPLOADED
    if UPLOADED:
        return render_template('ViewData.html',UPLOADED=UPLOADED,heading=dataFrame,data=np.array(dataFrame))
    return render_template('ViewData.html',UPLOADED=UPLOADED)


@app.route('/SplitData')
def SplitData():  # put application's code here
    global UPLOADED,SPLITTED,TRAINEDMODEL
    if UPLOADED:
        return render_template('SplitData.html',UPLOADED=UPLOADED)
        SPLITTED = False
        TRAINEDMODEL = False
    else:
        return render_template('SplitData.html',UPLOADED=UPLOADED)

@app.route('/SplitDataByPercentage',methods=['POST','GET'])
def SplitDataByPercentage():
    global UPLOADED,SPLITTED,TRAINEDMODEL,x_train,x_test,y_train,y_test,dataFrame,output
    if request.method == 'POST':
        percentage = request.form.get('percentage')
        output = request.form.get('output')
        if percentage=="" or output=="":
            return render_template('SplitData.html',UPLOADED=UPLOADED,SPLITTED=SPLITTED)
        percentage = float(percentage)/100
        output_data = dataFrame[[output]]
        input_data = dataFrame.drop([output],axis=1)
        x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=percentage)
        SPLITTED = True
        print(x_train.count()[0],x_test.count()[0])
    return render_template('SplitData.html',UPLOADED=UPLOADED,SPLITTED=SPLITTED,xTrain=x_train.count()[0],xTest=x_test.count()[0])

@app.route('/TrainModelWithOption',methods=['POST','GET'])
def TrainModelWithOption():
    if request.method == 'POST':
        global SPLITTED, TRAINEDMODEL, x_train, x_test, y_train, y_test, dataFrame, MODEL, model
        model = request.form.get('algo')
        acc=0
        TRAINEDMODEL=True
        if model=="LogisticRegression":
            MODEL = LogisticRegression()
            MODEL.fit(x_train, y_train)
            acc=accuracy_score(y_test,MODEL.predict(x_test))

    return render_template('TrainModel.html',UPLOADED=UPLOADED,SPLITTED=SPLITTED,TRAINEDMODEL=TRAINEDMODEL,acc=acc,model=model)

@app.route('/TrainModel')
def TrainModel():  # put application's code here
    global UPLOADED,SPLITTED,TRAINEDMODEL
    if UPLOADED:
        if SPLITTED:
            return render_template('TrainModel.html',UPLOADED=UPLOADED,SPLITTED=SPLITTED)
            TRAINEDMODEL = True
        else:
            return render_template('TrainModel.html',UPLOADED=UPLOADED,SPLITTED=SPLITTED)
    else:
        return render_template('TrainModel.html',UPLOADED=UPLOADED)

@app.route('/Prediction')
def Prediction():  # put application's code here
    global UPLOADED,SPLITTED,TRAINEDMODEL,dataFrame,data
    if UPLOADED:
        if SPLITTED:
            if TRAINEDMODEL:
                data = list(dataFrame.columns)
                data.remove(output)
                return render_template('Prediction.html',UPLOADED=UPLOADED,SPLITTED=SPLITTED,TRAINEDMODEL=TRAINEDMODEL,data=data)
            else:
               return render_template('Prediction.html',UPLOADED=UPLOADED,SPLITTED=SPLITTED,TRAINEDMODEL=TRAINEDMODEL)
        else:
            return render_template('Prediction.html',UPLOADED=UPLOADED,SPLITTED=SPLITTED)
    else:
        return render_template('Prediction.html',UPLOADED=UPLOADED)

@app.route('/predictOutput',methods=['POST','GET'])
def predictOutput():
    if request.method == 'POST':
        global SPLITTED, TRAINEDMODEL, x_train, x_test, y_train, y_test, dataFrame, MODEL, model,output,data
        input = request.form.get('input').split(" ")[:11]
        predict = MODEL.predict([input])
        if predict==0:
            predict = "NO"
        else:
            predict = "YES"
    return render_template('Prediction.html', UPLOADED=UPLOADED, SPLITTED=SPLITTED, TRAINEDMODEL=TRAINEDMODEL,PREDICTED=True,
                           data=data,predict=predict)
if __name__ == '__main__':
    app.run(debug=True)
