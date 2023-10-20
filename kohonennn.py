import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

class KohonenNetwork:
    def __init__(self, input_size, grid_size, learning_rate=0.1, epochs=1000):
        self.input_size = input_size
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.initial_radius = 5
        self.weights = {}
        self.bmus = {} # best matching unit for each country

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                self.weights[(i,j)] = np.random.rand(1, input_size)


    def kohonenRule(self, nk, ri):
        neighbors = set()
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                n = (i, j) 
                distance = np.linalg.norm(np.array(n) - np.array(nk))
                if distance < ri:
                    neighbors.add(n)
        
        return neighbors
            
    def winningNeuron(self, datapoint):
        # Calculate the Euclidean distances between the weight vector and input vector
        minNorm = np.inf
        wn = (0,0)
        for (i, j), weight_vector in self.weights.items():
            #print("wheight_vector:", weight_vector)
            #print("datapoint:", datapoint)
           # if self.input_size == 1:
            #    norm = weight_vector - datapoint
            #else:
            norm = np.linalg.norm(weight_vector - datapoint)
            if norm < minNorm: 
                minNorm = norm 
                wn = (i, j)
        return wn
    
    def updateWeights(self, neighbors, datapoint):

        for n in neighbors:        
            self.weights[n] += self.learning_rate * (datapoint - self.weights[n])

    

    def train(self, dataset):
        for epoch in range(self.epochs):
           
            ri = self.initial_radius - (self.initial_radius - 1) * (epoch / self.epochs)
            self.learning_rate = 1/(epoch+1)

            for i in range(len(dataset)):
               
                datapoint = dataset[i]
                wn = self.winningNeuron(datapoint)
                self.bmus[i] = wn
                neighbors = self.kohonenRule(wn, ri)

                self.updateWeights(neighbors, datapoint)
            
        print("self.bmus:", self.bmus)

    def visualize(self, countryIds, label):
        # each country will have the color representing its neuron

        fig, ax = plt.subplots()
        uniqueNeurons = list(set(self.bmus.values()))


        numUniqueNeurons = len(uniqueNeurons)
        
        colors =  sns.color_palette("husl", n_colors=numUniqueNeurons)
        
        
        neuronToColor  = {}
        i = 0
        for color in colors:
            neuronToColor[uniqueNeurons[i]] = color        
            i += 1

        print(neuronToColor)
        
        countryPositions = []
        for id in self.bmus.keys():
            (x,y) = (np.random.randint(int(len(self.bmus)**(1/2)+1)), np.random.randint(int(len(self.bmus)**(1/2)+1)))
            while (x,y) in countryPositions:
                (x,y) = (np.random.randint(int(len(self.bmus)**(1/2)+1)), np.random.randint(int(len(self.bmus)**(1/2)+1)))     
            countryPositions.append((x,y))
   
            ax.scatter((x,y)[0], (x,y)[1], c=neuronToColor[self.bmus[id]])
            ax.annotate(countryIds[id], (x,y), textcoords="offset points", xytext=(0,10), ha='center')


        # Set labels and title
        ax.set_title(label)

        plt.show()




# Example usage:
if __name__ == '__main__':

    economicalSize = 3
    socialSize = 3
    geographicSize = 1

    grid_size = (5,5)
    learning_rate = 1
    epochs = 50*economicalSize

    dataset = []
    countryIds = []
    with open("europe.csv", 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            dataset.append(row[1:])
            countryIds.append(row[0])
    countryIds = countryIds[1:]
    sc = MinMaxScaler(feature_range=(0,1))
   
    geographicData = [[row[0]] for row in dataset]  # Select the first and third columns
    economicalData = [[row[1], row[2], row[4]] for row in dataset] 
    socialData = [[row[3], row[5], row[6]] for row in dataset] 

    # does scalling and normalization
    economicalData = sc.fit_transform(economicalData[1:]) 
    socialData = sc.fit_transform(socialData[1:])   
    geographicData = sc.fit_transform(geographicData[1:])
    print(geographicData)

    somEconomical = KohonenNetwork(economicalSize, grid_size, learning_rate, epochs)
    somEconomical.train(economicalData)
   # somEconomical.visualize(countryIds, "Economical Features")

    somSocial = KohonenNetwork(socialSize, grid_size, learning_rate, epochs)
    somSocial.train(socialData)
  #  somSocial.visualize(countryIds, "Social Features")

    somgeographic = KohonenNetwork(geographicSize, grid_size, learning_rate, epochs)
    somgeographic.train(geographicData)
    somgeographic.visualize(countryIds, "Geographic Features")



    # missing this 
    # Realizar un gráfico que muestre las distancias promedio entre neuronas vecinas. 
    # Analizar la cantidad de elementos que fueron asociados a cada neurona.
