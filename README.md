# Cataract-Grading-using-Smartphone-Images
A cataract is an age-related eye disease and is one of the main ophthalmological public health problems in developed and developing countries. Early detection of cataracts is necessary to preserve sight and prevent the increase in blindness due to cataracts worldwide. Lacking eye clinicians and slit lamp cameras in poor and rural areas are the main causes of the cataract's late diagnoses. The recent research in this field indicates that it is possible to screen cataracts using image processing. As smartphones become universal in most urban areas, cataract self-screening with smartphones removes the limitations like cataract screening cost and travel/time burdens for patients. Accordingly, a novel computer-aided automatic cataract grading method is presented in the current project to detect various cataract stages, including normal, early, pre-mature, and mature cataracts, from the digital camera images. The IIITD Cataract Mobile Periocular (CMP) dataset was used as the cataractous and normal data images in the current study. This dataset contains periocular images, including ocular regions such as the eyebrow, pupil, sclera vasculature, iris, and pupil. These images are captured in the unconstrained condition such as uncontrolled illumination, complex background, and geometric distortions and mostly have non-frontal view poses. The current dissertation addresses smartphone-based cataract grading by proposing a method to classify the periocular eye regions into four classes of normal, early, pre-mature and mature cataracts on deep features using Convolutional Neural Networks (CNNs). We designed and proposed a four-layer CNN for cataract grading of the IIITD detected eye regions in the first procedure. In the second procedure, three pre-trained ConvNets, including VGG-16, Inception V3, and ResNet-101, were fine-tuned on the target dataset. In the last procedure, to evaluate the classification technique with the standard supervised classifiers, the extracted features by the ResNet-101 pre-trained network were fed into the Support Vector Machine (SVM) classifier for cataract grading. The experimental results show that end-to-end \gls{ResNet}-101 with the accuracy rate of 89.62 \% outperforms the four-layer CNN, VGG-16, Inception V3, and ResNet-101+SVM with the mean accuracy of 84.67\%, 87.64\%, 84.67\%, and 87.14\% respectively. Moreover, according to all the calculated evaluation metrics such as precision, recall, sensitivity, specificity, and also F-measure, which is the trade-off between recall and precision, the results show that for each class, ResNet-101 outperforms the other models and has a better grading result for IIITD with the imbalanced number of images for each class.


A guideline for the yploaded codes:

main1.py  
Using designed 4-layer CNN for grading the cataractous lenses in extracted eyes

main2.py
extracted eyes + keras aumentatgion + cnn


main4.py
agumented data + pre-trainded VGG16

main5.py
agumented data + inceptionv3

main7.py
agumented data + RezNet101


