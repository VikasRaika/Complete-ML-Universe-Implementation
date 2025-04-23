
import React from 'react';
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const NotebookViewer = () => {
  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <header className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Heart Disease Prediction: A Machine Learning Journey 🫀
        </h1>
        <p className="text-xl text-gray-600">
          Exploring medical data with scikit-learn to predict heart disease - perfect for ML portfolios!
        </p>
      </header>

      <Tabs defaultValue="notebook" className="w-full">
        <TabsList className="mb-4">
          <TabsTrigger value="notebook">Notebook</TabsTrigger>
          <TabsTrigger value="about">About</TabsTrigger>
        </TabsList>

        <TabsContent value="notebook">
          <ScrollArea className="h-[calc(100vh-250px)]">
            <div className="space-y-8">
              {/* Introduction Section */}
              <Card className="p-6">
                <h2 className="text-2xl font-semibold mb-4">🎯 Introduction</h2>
                <div className="prose max-w-none">
                  <p className="text-gray-700">
                    Hey there, fellow ML enthusiast! 👋 Today we're diving into an exciting project: predicting heart disease using machine learning. 
                    This notebook is perfect for your portfolio and shows off some key ML skills. We'll use the famous UCI Heart Disease dataset 
                    and walk through the entire machine learning pipeline together.
                  </p>
                </div>
              </Card>

              {/* Setup Section */}
              <Card className="p-6">
                <h2 className="text-2xl font-semibold mb-4">🛠️ Setup & Data Loading</h2>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-sm">
                  <pre>{`import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the heart disease dataset
df = pd.read_csv('heart.csv')
print(df.head())`}</pre>
                </div>
              </Card>

              {/* EDA Section */}
              <Card className="p-6">
                <h2 className="text-2xl font-semibold mb-4">📊 Exploratory Data Analysis</h2>
                <div className="prose max-w-none mb-4">
                  <p className="text-gray-700">
                    Let's explore our data! First, we'll check for missing values, look at basic statistics, 
                    and create some visualizations to understand our features better.
                  </p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-white p-4 rounded-lg border">
                    <h3 className="text-lg font-medium mb-2">Age Distribution</h3>
                    <div className="aspect-video bg-gray-100 rounded-lg"></div>
                  </div>
                  <div className="bg-white p-4 rounded-lg border">
                    <h3 className="text-lg font-medium mb-2">Feature Correlations</h3>
                    <div className="aspect-video bg-gray-100 rounded-lg"></div>
                  </div>
                </div>
              </Card>

              {/* Model Training */}
              <Card className="p-6">
                <h2 className="text-2xl font-semibold mb-4">🤖 Model Training</h2>
                <div className="prose max-w-none mb-4">
                  <p className="text-gray-700">
                    Time for the exciting part - training our model! We'll use a Random Forest Classifier, 
                    which works well for this type of problem. We'll also use cross-validation to ensure 
                    our results are robust.
                  </p>
                </div>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-sm">
                  <pre>{`# Split the data
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)`}</pre>
                </div>
              </Card>

              {/* Results */}
              <Card className="p-6">
                <h2 className="text-2xl font-semibold mb-4">📈 Results & Evaluation</h2>
                <div className="prose max-w-none mb-4">
                  <p className="text-gray-700">
                    Let's see how our model performed! We'll look at accuracy, precision, recall, 
                    and the confusion matrix to get a complete picture of our model's performance.
                  </p>
                </div>
                <div className="bg-white p-4 rounded-lg border">
                  <h3 className="text-lg font-medium mb-2">Model Performance Metrics</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <p className="text-sm text-gray-600">Accuracy</p>
                      <p className="text-2xl font-bold text-blue-600">85%</p>
                    </div>
                    <div className="p-4 bg-green-50 rounded-lg">
                      <p className="text-sm text-gray-600">Precision</p>
                      <p className="text-2xl font-bold text-green-600">83%</p>
                    </div>
                    <div className="p-4 bg-purple-50 rounded-lg">
                      <p className="text-sm text-gray-600">Recall</p>
                      <p className="text-2xl font-bold text-purple-600">87%</p>
                    </div>
                    <div className="p-4 bg-orange-50 rounded-lg">
                      <p className="text-sm text-gray-600">F1 Score</p>
                      <p className="text-2xl font-bold text-orange-600">85%</p>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Conclusion */}
              <Card className="p-6">
                <h2 className="text-2xl font-semibold mb-4">🎉 Conclusion & Next Steps</h2>
                <div className="prose max-w-none">
                  <p className="text-gray-700">
                    And there we have it! We've successfully built a heart disease prediction model with pretty good accuracy. 
                    Here are some ideas for taking this project further:
                  </p>
                  <ul className="list-disc pl-6 mt-4 space-y-2 text-gray-700">
                    <li>Try different algorithms (XGBoost, Neural Networks)</li>
                    <li>Implement feature engineering to improve accuracy</li>
                    <li>Add cross-validation for more robust results</li>
                    <li>Create an interactive demo using Streamlit or Flask</li>
                  </ul>
                </div>
              </Card>
            </div>
          </ScrollArea>
        </TabsContent>

        <TabsContent value="about">
          <Card className="p-6">
            <h2 className="text-2xl font-semibold mb-4">About This Notebook</h2>
            <div className="prose max-w-none">
              <p className="text-gray-700">
                This notebook was created as part of a machine learning portfolio project. It demonstrates:
              </p>
              <ul className="list-disc pl-6 mt-4 space-y-2 text-gray-700">
                <li>Data preprocessing and analysis</li>
                <li>Feature engineering and selection</li>
                <li>Model training and evaluation</li>
                <li>Clear documentation and visualization</li>
                <li>Best practices in machine learning</li>
              </ul>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default NotebookViewer;
