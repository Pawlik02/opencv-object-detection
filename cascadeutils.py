import os

print('Podaj nazwę folderu: ', end='')
dir_name = input()

file = open(f'haar/{dir_name}/negative.txt', 'w')
for filename in os.listdir(f'haar/{dir_name}/negative'):
    file.write('negative/' + filename + '\n')

file.close()

# Zaznaczanie obiektów
# /opt/opencv3/bin/opencv_annotation --annotations=dlugopisy/positive.txt --images=dlugopisy/positive/ --maxWindowHeight=100 --resizeFactor=5

# Stworzenie pliku .vec z którego potem algorytm się uczy
# /opt/opencv3/bin/opencv_createsamples -info dlugopisy/positive.txt -w 24 -h 24 -num 1000 -vec dlugopisy/positive.vec

# Uczenie algorytmem haar
# /opt/opencv3/bin/opencv_traincascade -data dlugopisy/cascade -vec dlugopisy/positive.vec -bg dlugopisy/negative.txt -w 24 -h 24 -numPos 100 -numNeg 200 -numStages 12

# Uczenie algorytmem lbp
# /opt/opencv3/bin/opencv_traincascade -featureType LBP -data dlugopisy/cascade -vec dlugopisy/positive.vec -bg dlugopisy/negative.txt -w 24 -h 24 -numPos 100 -numNeg 200 -numStages 12