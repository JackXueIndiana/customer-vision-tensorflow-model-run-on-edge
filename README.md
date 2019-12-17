# customer-vision-tensorflow-model-run-on-edge
This article shows how to use Azure Cognitive Services Customer Vision service to train an image bi-class classifier and the export the model to an IoT edge (a PC) to run the model locally.

Step 1. Create a Cusomter Vision account
Step 2. Under the account create a Project for Training
Step 3. Upload two groups of images and label them as "0" for negative and "1" for positive.
Step 4. Train the classifier
Step 5. Export the trained model in TensorFlow format in a zip file.
Step 6. Unzip the file in your PC
Step 7. Run the following code to classify an image.
The code is from https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/export-model-python
