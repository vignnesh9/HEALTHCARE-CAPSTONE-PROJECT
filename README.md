# HEALTHCARE-CAPSTONE-PROJECT
PROBLEM STATEMENT:
	ICMR wants to analyze different types of cancers, such as breast cancer, renal cancer, colon cancer, lung cancer, and prostate cancer becoming a cause of worry in recent years.
	 They would like to identify the probable cause of these cancers in terms of genes responsible for each cancer type. 
	This would lead us to early identification of each type of cancer reducing the fatality rate.
DATASET DETAILS:
	The input dataset contains 802 samples for the corresponding 802 people who have been detected with different types of cancer.
	 Each sample contains expression values of more than 20K genes. Samples have one of the types of tumors: BRCA, KIRC, COAD, LUAD, and PRAD

WEEK 1:- EXPLORATORY DATA ANALYSIS
1.	Exploratory Data Analysis:
2.	Merge both the datasets.
3.	Plot the merged dataset as a hierarchically-clustered heatmap.
4.	Perform Null-hypothesis testing.
	Shapiro test
	The null hypothesis for the Shapiro-Wilk test is that a variable is normally distributed in some population. 
	A different way to say the same is that a variable's values are a simple random sample from a normal distribution. 
	As a rule of thumb, we reject the null hypothesis if p < 0.05
	k2test - In statistics, D'Agostino's K2 test, named for Ralph D'Agostino, is a goodness-of-fit measure of departure from normality, that is the test aims to establish whether or not the given sample comes from a normally distributed population
	Dimensionality Reduction
	Each sample has expression values for around 20K genes. 
	However, it may not be necessary to include all 20K genes expression values to analyze each cancer type. 
	Therefore, we will identify a smaller set of attributes which will then be used to fit multiclass classification models. 
	So, the first task targets the dimensionality reduction using various techniques such as, PCA, LDA, and t-SNE. Input:
	 Complete dataset including all genes (20531) Output: Selected Genes from each dimensionality reduction method
PERFORM 
	PCA with n_components=2
	Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.
	Reducing the number of variables of a data set naturally comes at the expense of accuracy, but the trick in dimensionality reduction is to trade a little accuracy for simplicity. 
	Because smaller data sets are easier to explore and visualize and make analyzing data much easier and faster for machine learning algorithms without extraneous variables to process.
  
	PCA with n_components=.995

	Dimensionality reduction using TSNE
	T-SNE is a tool to visualize high-dimensional data. It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. 
	t-SNE has a cost function that is not convex, i.e. with different initializations we can get different results.
	Dimensionality reduction using LDA
	Linear Discriminant Analysis, or LDA for short, is a predictive modeling algorithm for multi-class classification.
	 It can also be used as a dimensionality reduction technique, providing a projection of a training dataset that best separates the examples by their assigned class. 
	The ability to use Linear Discriminant Analysis for dimensionality reduction often surprises most practitioners.

WEEK 2:- CLUSTERING GENES AND SAMPLES:
	Our next goal is to identify groups of genes that behave similarly across samples and identify the distribution of samples corresponding to each cancer type. Therefore, this task focuses on applying various clustering techniques, e.g., k-means, hierarchical, and mean-shift clustering, on genes and samples.
	First, apply the given clustering technique on all genes to identify:
	Genes whose expression values are similar across all samples
	Genes whose expression values are similar across samples of each cancer type
	Next, apply the given clustering technique on all samples to identify:
	Samples of the same class (cancer type) which also correspond to the same cluster
	Samples identified to be belonging to another cluster but also to the same class (cancer type)


	KMEANS Clustering with PCA = .995
	build classification models
	Building Classification Model(s) with Feature Selection:
Our final task is to build a robust classification model(s) for identifying each type of cancer.
	Sub-tasks:
	Build a classification model(s) using multiclass SVM, Random Forest, and Deep Neural Network to classify the input data into five cancer types
	Apply the feature selection algorithms, forward selection, and backward elimination to refine selected attributes (selected in Task-2) using the classification model from the previous step
	Validate the genes selected from the last step using statistical significance testing (t-test for one vs. all and F-test)
	Build decision tree clasifier
	Decision Tree is a Supervised Machine Learning Algorithm that uses a set of rules to make decisions, similarly to how humans make decisions. 
	One way to think of a Machine Learning classification algorithm is that it is built to make decisions. You usually say the model predicts the class of the new, never-seen-before input but, behind the scenes, the algorithm has to decide which class to assign


	SVM
	Support vector machine algorithm is used to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points


	Random Forest
	Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble.
	 Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction. 
	The fundamental concept behind random forest is a simple but powerful one — the wisdom of crowds. 
	In data science speak, the reason that the random forest model works so well is: A large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models.




	Naive Bayes Classifier
	A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. The crux of the classifier is based on the Bayes theorem.
	Bayes Theorem:
	Using Bayes theorem, we can find the probability of A happening, given that B has occurred. Here, B is the evidence and A is the hypothesis. The assumption made here is that the predictors/features are independent. That is presence of one particular feature does not affect the other. Hence it is called naive

	KNN Classifier
	K-nearest neighbors (KNN) algorithm is a type of supervised ML algorithm which can be used for both classification as well as regression predictive problems. However, it is mainly used for classification predictive problems in industry. The following two properties would define KNN well

	One way F test
	DNN
	The neural network needs to learn all the time to solve tasks in a more qualified manner or even to use various methods to provide a better result.
	 When it gets new information in the system, it learns how to act accordingly to a new situation.
	Learning becomes deeper when tasks you solve get harder. 
	Deep neural network represents the type of machine learning when the system uses many layers of nodes to derive high-level functions from input information.
	 It means transforming the data into a more creative and abstract component.
	In order to understand the result of deep learning better, let's imagine a picture of an average man. Although you have never seen this picture and his face and body before, you will always identify that it is a human and differentiate it from other creatures. 
	This is an example of how the deep neural network works. 
	Creative and analytical components of information are analyzed and grouped to ensure that the object is identified correctly. 
	These components are not brought to the system directly, thus the ML system has to modify and derive them.







	Define the model
	The ReLU function is f(x)=max(0,x). 
	Usually this is applied element-wise to the output of some other function, such as a matrix-vector product. In MLP usages, rectifier units replace all other activation functions except perhaps the readout layer.
	 But I suppose you could mix-and-match them if you'd like. One way ReLUs improve neural networks is by speeding up training. 
	The gradient computation is very simple (either 0 or 1 depending on the sign of x). 
	Also, the computational step of a ReLU is easy: any negative elements are set to 0.0 -- no exponentials, no multiplication or division operations. 
	Gradients of logistic and hyperbolic tangent networks are smaller than the positive portion of the ReLU. 
	This means that the positive portion is updated more rapidly as training progresses. However, this comes at a cost. 
	The 0 gradient on the left-hand side is has its own problem, called "dead neurons," in which a gradient update sets the incoming values to a ReLU such that the output is always zero; modified ReLU units such as ELU (or Leaky ReLU etc.) can minimize this. Source : StackExchange

	Evaluate the model
	Plot History
