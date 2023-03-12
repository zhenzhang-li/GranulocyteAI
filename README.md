# GranulocyteAI
SaGAN+SwinT- Constructed High-Confidence Automatic Morphology Examination App for Granulocytes.
Summary：
The recognition and counting of granulocytes can help doctors diagnose patients and make prescriptions. Conventional microscopic detection relies strongly on images from static databases for comparative discrimination, which can lead to a relatively high probability of misidentification. Deep learning allows feature extraction of subtle changes in the image and initiates a rapid, convenient, automated, and accurate clinical diagnosis of granulocytes. Here, we propose a hands-free application (App) to construct an automatic high-confidence quantitative measurement of granulocytes in the clinical laboratory. The integration of Spatial Attention GAN (SaGAN) and Swin Transformer (SwinT) model is used to effectively identify and count the four types of granulocytes, and the accuracy rate reached up to 99.48%. The clinical validation accuracy of the App is 99.49%. In a human-machine comparison, the app is more accurate than a doctor with 15 years' experience. This study provides an integrated method of AI capable of identifying and classifying granulocytes, thereby reducing labour and time costs, and assisting physicians in making rapid, interpretable, and accurate diagnoses. 

Figure 1. Overall framework of the model, three of which involve counting (I, II and III)：
<img width="415" alt="image" src="https://user-images.githubusercontent.com/78481822/224525978-ff9b4e17-68bf-475a-b2be-cd923d806735.png">

Figure 2. Model training framework diagram：
<img width="416" alt="image" src="https://user-images.githubusercontent.com/78481822/224526054-63f15209-c221-48a1-bc62-717e351120e5.png">

Figure 3. Schematic diagram of granulocyte real-time detection App:
<img width="416" alt="image" src="https://user-images.githubusercontent.com/78481822/224526015-e41b544a-5f6a-42d4-b6e8-f33c9f295522.png">
