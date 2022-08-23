import numpy as np
import pickle
import copy

class MultiClassLinearClassifier():
    """
    Multi class linear perceptron 
    """
    def __init__(self):
        """
        Defines number of classes (self.letters) and number of fetures (self.d)
        """
        self.letters = 26
        self.d = 8256
        self.w = None
        self.alphabet = list('abcdefghijklmnopqrstuvwxyz')
    
    def predict(self, x, weights = None):
        """
        Predicts letter.
        Makes dot prouct of input feature vector x with all columns of 
        matrix w. Than argmax selects highest value from dot pruct which
        is equal to most probable class predicted
        """
        if weights is None:
            weights = self.w

        return np.argmax(np.dot(weights.T,np.append(x,1)))

    def update_weights(self, w, x, prediction, true_letter):
        """
        Updates weights matrix. Ground truth is added to 
        w and wrong prediction is subtracted. There is 1 appended to x
        because matrix w have Bias (vector b) as last row.
        """
        if not prediction == true_letter:
            w[:,true_letter] += np.append(x,1)
            w[:,prediction] -= np.append(x,1)
        return w

    def letter_to_int(self, letter):
        """
        Converts letter to number. a = 0, z = 25
        """
        return self.alphabet.index(letter)
    
    def compute_error(self, X, Y):
        """
        Computes prediction error on dataset X,Y
        """
        results = []
        M = 0
        m = len(X)
        for img_id in range(m):
            results.append([])
            for letter_id, letter in enumerate(Y[img_id]):
                M += 1
                x = X[img_id][:,letter_id]
                true_letter = self.letter_to_int(letter)
                prediction = self.predict(x)
                if prediction == true_letter:
                    results[img_id].append(0)
                else:
                    results[img_id].append(1)
        seq_err = 0
        char_err = 0
        for res in results:
            if sum(res) > 0:
                seq_err += 1
                char_err += sum(res)
        print("Error on sequence: %.2f, Error on characters: %.2f"%(seq_err/m, char_err/M))
        
    def fit(self, X, Y):
        """
        Fit pereptron agorithm on X,Y dataset
        """
        n = len(X)
        w = np.zeros((self.d+1,self.letters))
        
        print('Fitting: ', end = '')
        iteration = 0
        while(True):
            old_w = copy.deepcopy(w)
            for img_id in range(n):
                for letter_id, letter in enumerate(Y[img_id]):
                    x = X[img_id][:,letter_id]
                    true_letter = self.letter_to_int(letter)
                    prediction = self.predict(x, w)
                    w = self.update_weights(w, x, prediction, true_letter)
                    
            if iteration%5==0:
                print('#', end = '')

            iteration += 1
            fitted = True
            for i in range(self.letters):
                if not np.array_equal(w[i], old_w[i]):
                    fitted = False
                    break
                    
            if fitted:
                break
                
        print('\nFitting done')
        self.w = w


class LinearStructuredOutputClassifier():
    def __init__(self):
        """
        Defines number of classes (self.letters) and number of fetures (self.d)
        """
        self.letters = 400
        self.d = 55
        self.w = np.zeros((self.d+1, self.letters))
        self.g = np.zeros((self.letters, self.letters))
        self.alphabet = list('abcdefghijklmnopqrstuvwxyz')

    def predict(self, word):
        """
        Predicts letter.
        Makes dot prouct of input feature vector x with all columns of 
        matrix w. Than argmax selects highest value from dot pruct which
        is equal to most probable class predicted
        """
        weights = self.w
        
        L = word.shape[1]
        F = np.zeros((self.letters, L))
        GF_id = np.zeros((self.letters, L)).astype(int)
        prediction = [0]*L
        
        # Forward pass
        F[:,0] = np.dot(weights.T,np.append(word[:,0],1))
        for i in range(1,L):
            q = np.dot(weights.T,np.append(word[:,i],1))
            g = self.g.T + F[:,i-1]
            F[:,i] = q.T + np.max(g.T, axis=0)
            GF_id[:,i] = np.argmax(g.T, axis=0)
            
        # Backward pass
        prediction[L-1] = np.argmax(F[:,L-1])
        for i in range(L-2,-1,-1):
            prediction[i] = GF_id[prediction[i+1],i+1]
        
        return prediction
    
    def update_weights(self, x, y, prediction):
        """
        Updates weights matrix. Ground truth is added to 
        w and wrong prediction is subtracted. There is 1 appended to x
        because matrix w have Bias (vector b) as last row.
        """
        L = x.shape[1]
        for letter_id in range(L):
            if not prediction[letter_id] == y[letter_id]:
                mse = (prediction[letter_id] - y[letter_id])**2
                correction = 2500/mse
                correct_id = y[L-1]
                self.w[:,correct_id] += np.append(x[:,L-1],correction)
                self.w[:,prediction[L-1]] -= np.append(x[:,L-1],correction)

                for i in range(L-1):
                    correct_id = y[i]
                    next_correct_id = y[i+1]
                    self.w[:,correct_id] += np.append(x[:,i],correction)
                    self.w[:,prediction[i]] -= np.append(x[:,i],correction)
                    self.g[correct_id,next_correct_id] += correction
                    self.g[prediction[i],prediction[i+1]] -= correction
        # TODO:
        # There should be some differnet update based on MSE not just +- 1


    def update_weights_test(self, x, y, prediction):
        """
        Second update weights. I still think that working one is 
        wrong. I dont like the fact that both w and g matrix are updated if single letter 
        is wrong. But I am most probably wrong since the original update_weights is
        the only ona that converges.
        """
        L = x.shape[1]
        for letter_id in range(L):
            if not prediction[letter_id] == y[letter_id]:
                mse = (prediction[letter_id] - y[letter_id])**2
                correction = 2500/mse
                correct_id = y[letter_id]
                self.w[:,correct_id] += np.append(x[:,letter_id],correction)
                self.w[:,prediction[letter_id]] -= np.append(x[:,letter_id],correction)

                for i in range(letter_id, L-1):
                    correct_id = y[i]
                    next_correct_id = y[i+1]
                    self.g[correct_id,next_correct_id] += correction
                    self.g[prediction[i],prediction[i+1]] -= correction
            
    def compute_error(self, X, Y, silent=False):
        """
        Computes prediction error on dataset X,Y
        """
        M = 0
        m = len(X)

        total = 0

        for img_id in range(m):
            days.append([])
            prediction = self.predict(X[img_id])
            for letter_id, letter in enumerate(Y[img_id]):
                M += 1
                predicted_letter = prediction[letter_id]
                mse = (predicted_letter - letter)**2
                total += mse

        total_mse = total/M

        if not silent:
            print("Total MSE: %.2f"%(total_mse))
        
        return total_mse, 0
    
    def fit(self, X, Y, max_iter=1000, silent=True):
        """
        Fit pereptron agorithm on X,Y dataset
        """
        n = len(X)
        
        print('Fitting: ', end = '')
        iteration = 0
        while(True):
            old_w = copy.deepcopy(self.w)
            for img_id in range(n):
                prediction = self.predict(X[img_id])
                self.update_weights(X[img_id], Y[img_id], prediction)
                (seq_err, char_err) = self.compute_error(X, Y, silent)

            if iteration%5==0:
                print('#', end = '')

            iteration += 1                  
            mse = self.compute_error(X, Y, silent)
            if mse == 0.0:
                break

            if iteration > max_iter:
                break

        print('\nFitting done')
