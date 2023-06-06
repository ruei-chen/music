import sys
import os
import random
import shutil
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QLabel, QApplication, QWidget, QPushButton, QStackedWidget, QCheckBox, QFileDialog)
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt, QTimer
import glob
import numpy
import pygame
from music21 import converter, instrument, note, chord, stream
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import ModelCheckpoint


class Page1(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        image_label = QLabel(self)
        pixmap = QPixmap('images/image.jpg')
        image_label.setPixmap(pixmap)

        self.button1 = QPushButton('Get Started',self)
        self.button1.setGeometry(QtCore.QRect(400, 500, 300, 100))
        self.button1.setStyleSheet('background-color: rgba(0, 0, 0, 0); font-size: 24px; color: white; font-weight: 900')
        self.button1.clicked.connect(self.goToPage2)

        self.button2 = QPushButton('Review',self)
        self.button2.setGeometry(QtCore.QRect(750, 500, 300, 100))
        self.button2.setStyleSheet('background-color: rgba(0, 0, 0, 0); font-size: 24px; color: white; font-weight: 900')
        self.button2.clicked.connect(self.goToPage6)

        self.startTextAnimation()

    def startTextAnimation(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateButtonStyle)
        self.timer.start(750)  # Update the style every 7000 milliseconds (0.7 second)

    def updateButtonStyle(self):
        current_color = self.button1.palette().color(self.button1.foregroundRole())
        if current_color == Qt.white:
            self.button1.setStyleSheet('background-color: rgba(0, 0, 0, 0); font-size: 24px; color: transparent; font-weight: 900')
            self.button2.setStyleSheet('background-color: rgba(0, 0, 0, 0); font-size: 24px; color: white; font-weight: 900')

        else:
            self.button1.setStyleSheet('background-color: rgba(0, 0, 0, 0); font-size: 24px; color: white; font-weight: 900')
            self.button2.setStyleSheet('background-color: rgba(0, 0, 0, 0); font-size: 24px; color: transparent; font-weight: 900')

    def goToPage2(self):
        # 觸發轉移到Page2
        main_widget.setCurrentIndex(1)  # 將索引設置為1，即第二個頁面

    def goToPage6(self):
        # 觸發轉移到Page2
        main_widget.setCurrentIndex(7)  # 將索引設置為1，即第二個頁面

class Page2(QWidget):
    def __init__(self):
        super().__init__()
        self.change = 0
        self.initUI()

    def initUI(self):
        self.background0 = QWidget(self) #background
        self.background0.setGeometry(0,0,1080,608) #(水平位置,垂直位置,寬,高)
        self.background0.setStyleSheet("background-color: #B3D9D9")

        self.background1 = QWidget(self) #background
        self.background1.setGeometry(108,152,210,250) #(水平位置,垂直位置,寬,高)
        self.background1.setStyleSheet("background-color: #F0F0F0; border-radius: 100%;")

        self.background2 = QWidget(self) #background
        self.background2.setGeometry(438,152,210,250) #(水平位置,垂直位置,寬,高)
        self.background2.setStyleSheet("background-color: #F0F0F0; border-radius: 100%;")

        self.background3 = QWidget(self) #background
        self.background3.setGeometry(758,152,210,250) #(水平位置,垂直位置,寬,高)
        self.background3.setStyleSheet("background-color: #F0F0F0; border-radius: 100%;")

        self.background4 = QWidget(self) #background
        self.background4.setGeometry(470,528,150,45) #(水平位置,垂直位置,寬,高)
        self.background4.setStyleSheet("background-color: #FFDCB9; border-radius: 20px;")
        
        self.button1 = QPushButton('GO BACK ',self)
        self.button1.setGeometry(QtCore.QRect(450, 525, 200, 50))
        self.button1.setStyleSheet('background-color: transparent; font-size: 24px; color: #003D79; font-weight: 900')
        self.button1.clicked.connect(self.goToPage1)

        self.button2 = QPushButton('Lo-Fi Hip Hop',self)
        self.button2.setGeometry(QtCore.QRect(120, 250, 200, 50))
        self.button2.setStyleSheet('background-color: transparent; font-size: 24px; color: #003D79; font-weight: 900')
        self.button2.clicked.connect(self.goToPage3_1)

        self.button3 = QPushButton('Classical ',self)
        self.button3.setGeometry(QtCore.QRect(450, 250, 200, 50))
        self.button3.setStyleSheet('background-color: transparent; font-size: 24px; color: #003D79; font-weight: 900')
        self.button3.clicked.connect(self.goToPage3_2)

        self.button4 = QPushButton('POP Music',self)
        self.button4.setGeometry(QtCore.QRect(770, 250, 200, 50))
        self.button4.setStyleSheet('background-color: transparent; font-size: 24px; color: #003D79; font-weight: 900')
        self.button4.clicked.connect(self.goToPage3_3)
        
        self.button5 = QPushButton('remodule',self)
        self.button5.setGeometry(QtCore.QRect(930, 540, 100, 50))
        self.button5.setStyleSheet('background-color: transparent; font-size: 24px; color: #EA7500; font-weight: 600; text-decoration: underline;')
        self.button5.clicked.connect(self.goToPage5)

        
    def goToPage1(self):
        # 觸發轉移到Page1
        main_widget.setCurrentIndex(0)  # 將索引設置為0，即第一個頁面

    def goToPage3_1(self):
        main_widget.setCurrentIndex(2)

    def goToPage3_2(self):
        main_widget.setCurrentIndex(3)

    def goToPage3_3(self):
        main_widget.setCurrentIndex(4) 
        
    def goToPage5(self):
        main_widget.setCurrentIndex(6)

class Page3_1(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.original_pixmap1 = QPixmap('images/lofi1.jpg')
        self.original_pixmap2 = QPixmap('images/lofi2.jpg')

        # 設定目標寬度和高度
        target_width = 540
        target_height = 608
        
        self.pixmap1 = self.original_pixmap1.scaled(target_width, target_height)
        self.pixmap2 = self.original_pixmap2.scaled(target_width, target_height)

        self.image_label1 = QLabel(self)
        self.image_label1.setPixmap(self.pixmap1)
        self.image_label1.setGeometry(0, 0, 540, 608)

        self.image_label2 = QLabel(self)
        self.image_label2.setPixmap(self.pixmap2)
        self.image_label2.setGeometry(540, 0, 540, 608)

        self.background1 = QWidget(self) #background
        self.background1.setGeometry(470,528,150,45) #(水平位置,垂直位置,寬,高)
        self.background1.setStyleSheet("background-color: #FFDCB9; border-radius: 20px;")
        
        self.button1 = QPushButton('GO BACK ',self)
        self.button1.setGeometry(QtCore.QRect(450, 525, 200, 50))
        self.button1.setStyleSheet('background-color: transparent; font-size: 22px; color: #003D79; font-weight: 900')
        self.button1.clicked.connect(self.goToPage2)

        self.button2 = QPushButton('Lo-Fi soothing',self)
        self.button2.setGeometry(QtCore.QRect(120, 490, 200, 50))
        self.button2.setStyleSheet('background-color: #BEBEBE; font-size: 24px; color: #003D79; font-weight: 500')
        self.button2.clicked.connect(self.lofi1goToPage4)

        self.button3 = QPushButton('Lo-Fi gentle',self)
        self.button3.setGeometry(QtCore.QRect(760, 490, 200, 50))
        self.button3.setStyleSheet('background-color: #BEBEBE; font-size: 24px; color: #003D79; font-weight: 500')
        self.button3.clicked.connect(self.lofi2goToPage4)
        
    def goToPage2(self):
        # 觸發轉移到Page2
        main_widget.setCurrentIndex(1)  # 將索引設置為1，即第一個頁面

    def lofi1goToPage4(self):
        notes = []
        for file in glob.glob("Lo-Fi Hip Hop/1/*.mid"):
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        sequence_length = 90
        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        n_vocab = len(pitchnames)
        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab)
        network_output = np_utils.to_categorical(network_output, num_classes=n_vocab)

        model = Sequential()
        model.add(LSTM(
            256,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128))
        model.add(Dense(128))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        
        directory = 'Lo-Fi Hip Hop/Lo-Fi Hip Hop1-weight/'

        # Get a list of all HDF5 files in the directory
        hdf5_files = [file for file in os.listdir(directory) if file.endswith('.hdf5')]
        # Randomly select a file
        selected_file = random.choice(hdf5_files)
        file_path = os.path.join(directory, selected_file)
        # Load the weights to each node
        model.load_weights(file_path)

        start = numpy.random.randint(0, len(network_input)-1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        pattern = network_input[start].tolist()
        prediction_output = []
        # generate 250 notes
        for _ in range(90):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)
            pattern = numpy.append(pattern, index)
            pattern = pattern[1:len(pattern)]

        offset = 0
        output_notes = []
        # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.4

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='test_output.mid')

        # 初始化Pygame
        pygame.init()

        # 設定輸入的MIDI檔案路徑和輸出的音訊檔案路徑
        midi_file = "test_output.mid"

        # 設定音訊參數
        sample_rate = 44100
        bit_depth = -16
        channels = 2

        # 設定音訊輸出
        pygame.mixer.init(sample_rate, bit_depth, channels)

        # 播放MIDI檔案
        pygame.mixer.music.load(midi_file)
        pygame.mixer.music.play()

        # 等待音樂播放結束
        while pygame.mixer.music.get_busy():
            pass

        # 停止音訊輸出
        pygame.mixer.quit()
        # 觸發轉移到Page4
        main_widget.setCurrentIndex(5)  # 將索引設置為5，即第四個頁面

    def lofi2goToPage4(self):
        notes = []
        for file in glob.glob("Lo-Fi Hip Hop/2/*.mid"):
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        sequence_length = 90
        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        n_vocab = len(pitchnames)
        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab)
        network_output = np_utils.to_categorical(network_output, num_classes=n_vocab)

        model = Sequential()
        model.add(LSTM(
            256,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128))
        model.add(Dense(128))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        
        directory = 'Lo-Fi Hip Hop/Lo-Fi Hip Hop2-weight/'

        # Get a list of all HDF5 files in the directory
        hdf5_files = [file for file in os.listdir(directory) if file.endswith('.hdf5')]
        # Randomly select a file
        selected_file = random.choice(hdf5_files)
        file_path = os.path.join(directory, selected_file)
        # Load the weights to each node
        model.load_weights(file_path)

        start = numpy.random.randint(0, len(network_input)-1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        pattern = network_input[start].tolist()
        prediction_output = []
        # generate 250 notes
        for _ in range(90):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)
            pattern = numpy.append(pattern, index)
            pattern = pattern[1:len(pattern)]

        offset = 0
        output_notes = []
        # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.4

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='test_output.mid')

        # 初始化Pygame
        pygame.init()

        # 設定輸入的MIDI檔案路徑和輸出的音訊檔案路徑
        midi_file = "test_output.mid"

        # 設定音訊參數
        sample_rate = 44100
        bit_depth = -16
        channels = 2

        # 設定音訊輸出
        pygame.mixer.init(sample_rate, bit_depth, channels)

        # 播放MIDI檔案
        pygame.mixer.music.load(midi_file)
        pygame.mixer.music.play()

        # 等待音樂播放結束
        while pygame.mixer.music.get_busy():
            pass

        # 停止音訊輸出
        pygame.mixer.quit()
        # 觸發轉移到Page4
        main_widget.setCurrentIndex(5)  # 將索引設置為5，即第四個頁面


class Page3_2(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.original_pixmap1 = QPixmap('images/Schubert.jpg')
        self.original_pixmap2 = QPixmap('images/Beethoven.jpg')

        # 設定目標寬度和高度
        target_width = 540
        target_height = 608
        
        self.pixmap1 = self.original_pixmap1.scaled(target_width, target_height)
        self.pixmap2 = self.original_pixmap2.scaled(target_width, target_height)

        self.image_label1 = QLabel(self)
        self.image_label1.setPixmap(self.pixmap1)
        self.image_label1.setGeometry(0, 0, 540, 608)

        self.image_label2 = QLabel(self)
        self.image_label2.setPixmap(self.pixmap2)
        self.image_label2.setGeometry(540, 0, 540, 608)

        self.background1 = QWidget(self) #background
        self.background1.setGeometry(470,528,150,45) #(水平位置,垂直位置,寬,高)
        self.background1.setStyleSheet("background-color: #FFDCB9; border-radius: 20px;")
        
        self.button1 = QPushButton('GO BACK ',self)
        self.button1.setGeometry(QtCore.QRect(450, 525, 200, 50))
        self.button1.setStyleSheet('background-color: transparent; font-size: 22px; color: #003D79; font-weight: 900')
        self.button1.clicked.connect(self.goToPage2)

        self.button2 = QPushButton('schubert',self)
        self.button2.setGeometry(QtCore.QRect(120, 490, 200, 50))
        self.button2.setStyleSheet('background-color: #BEBEBE; font-size: 24px; color: #003D79; font-weight: 500')
        self.button2.clicked.connect(self.Classical1goToPage4)

        self.button3 = QPushButton('beethoven',self)
        self.button3.setGeometry(QtCore.QRect(760, 490, 200, 50))
        self.button3.setStyleSheet('background-color: #BEBEBE; font-size: 24px; color: #003D79; font-weight: 500')
        self.button3.clicked.connect(self.Classical2goToPage4)
        
    def goToPage2(self):
        # 觸發轉移到Page2
        main_widget.setCurrentIndex(1)  # 將索引設置為1，即第一個頁面

    def Classical1goToPage4(self):
        notes = []
        for file in glob.glob("Classical/schubert/*.mid"):
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        sequence_length = 90
        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        n_vocab = len(pitchnames)
        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab)
        network_output = np_utils.to_categorical(network_output, num_classes=n_vocab)

        model = Sequential()
        model.add(LSTM(
            256,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128))
        model.add(Dense(128))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        
        directory = 'Classical/schubert-weight/'

        # Get a list of all HDF5 files in the directory
        hdf5_files = [file for file in os.listdir(directory) if file.endswith('.hdf5')]
        # Randomly select a file
        selected_file = random.choice(hdf5_files)
        file_path = os.path.join(directory, selected_file)
        # Load the weights to each node
        model.load_weights(file_path)

        start = numpy.random.randint(0, len(network_input)-1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        pattern = network_input[start].tolist()
        prediction_output = []
        # generate 250 notes
        for _ in range(90):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)
            pattern = numpy.append(pattern, index)
            pattern = pattern[1:len(pattern)]

        offset = 0
        output_notes = []
        # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.45

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='test_output.mid')

        # 初始化Pygame
        pygame.init()

        # 設定輸入的MIDI檔案路徑和輸出的音訊檔案路徑
        midi_file = "test_output.mid"

        # 設定音訊參數
        sample_rate = 44100
        bit_depth = -16
        channels = 2

        # 設定音訊輸出
        pygame.mixer.init(sample_rate, bit_depth, channels)

        # 播放MIDI檔案
        pygame.mixer.music.load(midi_file)
        pygame.mixer.music.play()

        # 等待音樂播放結束
        while pygame.mixer.music.get_busy():
            pass

        # 停止音訊輸出
        pygame.mixer.quit()
        # 觸發轉移到Page4
        main_widget.setCurrentIndex(5)  # 將索引設置為5，即第四個頁面

    def Classical2goToPage4(self):
        notes = []
        for file in glob.glob("Classical/beeth/*.mid"):
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        sequence_length = 90
        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        n_vocab = len(pitchnames)
        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab)
        network_output = np_utils.to_categorical(network_output, num_classes=n_vocab)

        model = Sequential()
        model.add(LSTM(
            256,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128))
        model.add(Dense(128))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        
        directory = 'Classical/beeth-weight/'

        # Get a list of all HDF5 files in the directory
        hdf5_files = [file for file in os.listdir(directory) if file.endswith('.hdf5')]
        # Randomly select a file
        selected_file = random.choice(hdf5_files)
        file_path = os.path.join(directory, selected_file)
        # Load the weights to each node
        model.load_weights(file_path)

        start = numpy.random.randint(0, len(network_input)-1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        pattern = network_input[start].tolist()
        prediction_output = []
        # generate 250 notes
        for _ in range(90):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)
            pattern = numpy.append(pattern, index)
            pattern = pattern[1:len(pattern)]

        offset = 0
        output_notes = []
        # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.45

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='test_output.mid')

        # 初始化Pygame
        pygame.init()

        # 設定輸入的MIDI檔案路徑和輸出的音訊檔案路徑
        midi_file = "test_output.mid"

        # 設定音訊參數
        sample_rate = 44100
        bit_depth = -16
        channels = 2

        # 設定音訊輸出
        pygame.mixer.init(sample_rate, bit_depth, channels)

        # 播放MIDI檔案
        pygame.mixer.music.load(midi_file)
        pygame.mixer.music.play()

        # 等待音樂播放結束
        while pygame.mixer.music.get_busy():
            pass

        # 停止音訊輸出
        pygame.mixer.quit()
        # 觸發轉移到Page4
        main_widget.setCurrentIndex(5)  # 將索引設置為5，即第四個頁面

class Page3_3(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.original_pixmap1 = QPixmap('images/Chainsmokers.jpg')
        self.original_pixmap2 = QPixmap('images/Ariana Grande.jpg')

        # 設定目標寬度和高度
        target_width = 540
        target_height = 608
        
        self.pixmap1 = self.original_pixmap1.scaled(target_width, target_height)
        self.pixmap2 = self.original_pixmap2.scaled(target_width, target_height)

        self.image_label1 = QLabel(self)
        self.image_label1.setPixmap(self.pixmap1)
        self.image_label1.setGeometry(0, 0, 540, 608)

        self.image_label2 = QLabel(self)
        self.image_label2.setPixmap(self.pixmap2)
        self.image_label2.setGeometry(540, 0, 540, 608)

        self.background1 = QWidget(self) #background
        self.background1.setGeometry(470,528,150,45) #(水平位置,垂直位置,寬,高)
        self.background1.setStyleSheet("background-color: #FFDCB9; border-radius: 20px;")
        
        self.button1 = QPushButton('GO BACK ',self)
        self.button1.setGeometry(QtCore.QRect(450, 525, 200, 50))
        self.button1.setStyleSheet('background-color: transparent; font-size: 22px; color: #003D79; font-weight: 900')
        self.button1.clicked.connect(self.goToPage2)

        self.button2 = QPushButton('Chainsmokers',self)
        self.button2.setGeometry(QtCore.QRect(120, 490, 200, 50))
        self.button2.setStyleSheet('background-color: #BEBEBE; font-size: 24px; color: #003D79; font-weight: 500')
        self.button2.clicked.connect(self.pop1goToPage4)

        self.button3 = QPushButton('Ariana Grande',self)
        self.button3.setGeometry(QtCore.QRect(760, 490, 200, 50))
        self.button3.setStyleSheet('background-color: #BEBEBE; font-size: 24px; color: #003D79; font-weight: 500')
        self.button3.clicked.connect(self.pop2goToPage4)
        
    def goToPage2(self):
        # 觸發轉移到Page2
        main_widget.setCurrentIndex(1)  # 將索引設置為1，即第二個頁面

    def pop1goToPage4(self):
        notes = []
        for file in glob.glob("pop/1/*.mid"):
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        sequence_length = 90
        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        n_vocab = len(pitchnames)
        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab)
        network_output = np_utils.to_categorical(network_output, num_classes=n_vocab)

        model = Sequential()
        model.add(LSTM(
            256,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128))
        model.add(Dense(128))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        
        directory = 'pop/pop1-weight/'

        # Get a list of all HDF5 files in the directory
        hdf5_files = [file for file in os.listdir(directory) if file.endswith('.hdf5')]
        # Randomly select a file
        selected_file = random.choice(hdf5_files)
        file_path = os.path.join(directory, selected_file)
        # Load the weights to each node
        model.load_weights(file_path)

        start = numpy.random.randint(0, len(network_input)-1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        pattern = network_input[start].tolist()
        prediction_output = []
        # generate 90 notes
        for _ in range(90):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)
            pattern = numpy.append(pattern, index)
            pattern = pattern[1:len(pattern)]

        offset = 0
        output_notes = []
        # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='test_output.mid')

        # 初始化Pygame
        pygame.init()

        # 設定輸入的MIDI檔案路徑和輸出的音訊檔案路徑
        midi_file = "test_output.mid"

        # 設定音訊參數
        sample_rate = 44100
        bit_depth = -16
        channels = 2

        # 設定音訊輸出
        pygame.mixer.init(sample_rate, bit_depth, channels)

        # 播放MIDI檔案
        pygame.mixer.music.load(midi_file)
        pygame.mixer.music.play()

        # 等待音樂播放結束
        while pygame.mixer.music.get_busy():
            pass

        # 停止音訊輸出
        pygame.mixer.quit()
        # 觸發轉移到Page4
        main_widget.setCurrentIndex(5)  # 將索引設置為5，即第四個頁面

    def pop2goToPage4(self):
        notes = []
        for file in glob.glob("pop/2/*.mid"):
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        sequence_length = 90
        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        n_vocab = len(pitchnames)
        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab)
        network_output = np_utils.to_categorical(network_output, num_classes=n_vocab)

        model = Sequential()
        model.add(LSTM(
            256,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128))
        model.add(Dense(128))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        
        directory = 'pop/pop2-weight/'

        # Get a list of all HDF5 files in the directory
        hdf5_files = [file for file in os.listdir(directory) if file.endswith('.hdf5')]
        # Randomly select a file
        selected_file = random.choice(hdf5_files)
        file_path = os.path.join(directory, selected_file)
        # Load the weights to each node
        model.load_weights(file_path)

        start = numpy.random.randint(0, len(network_input)-1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        pattern = network_input[start].tolist()
        prediction_output = []
        # generate 250 notes
        for _ in range(90):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)
            pattern = numpy.append(pattern, index)
            pattern = pattern[1:len(pattern)]

        offset = 0
        output_notes = []
        # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='test_output.mid')

        # 初始化Pygame
        pygame.init()

        # 設定輸入的MIDI檔案路徑和輸出的音訊檔案路徑
        midi_file = "test_output.mid"

        # 設定音訊參數
        sample_rate = 44100
        bit_depth = -16
        channels = 2

        # 設定音訊輸出
        pygame.mixer.init(sample_rate, bit_depth, channels)

        # 播放MIDI檔案
        pygame.mixer.music.load(midi_file)
        pygame.mixer.music.play()

        # 等待音樂播放結束
        while pygame.mixer.music.get_busy():
            pass

        # 停止音訊輸出
        pygame.mixer.quit()
        # 觸發轉移到Page4
        main_widget.setCurrentIndex(5)  # 將索引設置為5，即第四個頁面

class Page4(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.original_pixmap1 = QPixmap('images/music.jpg')

        # 設定目標寬度和高度
        target_width = 1080
        target_height = 608
        
        self.pixmap1 = self.original_pixmap1.scaled(target_width, target_height)

        self.image_label1 = QLabel(self)
        self.image_label1.setPixmap(self.pixmap1)
        self.image_label1.setGeometry(0, 0, 1080, 608)

        self.background1 = QWidget(self) #background
        self.background1.setGeometry(470,528,150,45) #(水平位置,垂直位置,寬,高)
        self.background1.setStyleSheet("background-color: #FFDCB9; border-radius: 20px;")

        self.background2 = QWidget(self) #background
        self.background2.setGeometry(60,500,120,45) #(水平位置,垂直位置,寬,高)
        self.background2.setStyleSheet("background-color: #7B7B7B; border-radius: 20px;")


        self.button1 = QPushButton('GO BACK ',self)
        self.button1.setGeometry(QtCore.QRect(450, 525, 200, 50))
        self.button1.setStyleSheet('background-color: transparent; font-size: 22px; color: #003D79; font-weight: 900')
        self.button1.clicked.connect(self.goToPage2)
        
        self.button2 = QPushButton('Save File', self)
        self.button2.setGeometry(QtCore.QRect(60, 470, 120, 100))
        self.button2.setStyleSheet('background-color: transparent; font-size: 22px; color: #FFFF37; font-weight: 900')
        self.button2.clicked.connect(self.saveFile)

        self.label = QLabel('格式:"example.mid"',self) #create label
        self.label.setGeometry(QtCore.QRect(40,535,300,50)) #(水平位置,垂直位置,寬,高)
        self.label.setStyleSheet('font-size: 18px; color: black; font-weight: 600')

    def saveFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getSaveFileName(self, "Save File", "", "MIDI Files (*.mid)", options=options)

        if filepath:
            original_filepath = "test_output.mid"
            try:
                shutil.copy(original_filepath, filepath)
                print("File saved successfully:", filepath)
            except Exception as e:
                print("Error saving file:", str(e))

    
    def goToPage2(self):
        # 觸發轉移到Page2
        main_widget.setCurrentIndex(1)  # 將索引設置為1，即第二個頁面

class Page5(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.background0 = QWidget(self) #background
        self.background0.setGeometry(0,0,1080,608) #(水平位置,垂直位置,寬,高)
        self.background0.setStyleSheet("background-color: #B3D9D9")

        self.background4 = QWidget(self) #background
        self.background4.setGeometry(470,528,150,45) #(水平位置,垂直位置,寬,高)
        self.background4.setStyleSheet("background-color: #FFDCB9; border-radius: 20px;")

        self.button1 = QPushButton('GO BACK ',self)
        self.button1.setGeometry(QtCore.QRect(450, 525, 200, 50))
        self.button1.setStyleSheet('background-color: transparent; font-size: 24px; color: #003D79; font-weight: 900')
        self.button1.clicked.connect(self.goToPage2)

        self.label1 = QLabel('Lo-Fi Hip Hop',self) #create label
        self.label1.setGeometry(QtCore.QRect(100,70,200,50)) #(水平位置,垂直位置,寬,高)
        self.label1.setStyleSheet('font-size: 32px; color: black; font-weight: 900')
        
        self.label2 = QLabel('Classical',self) #create label 
        self.label2.setGeometry(QtCore.QRect(485,70,200,50)) #(水平位置,垂直位置,寬,高)
        self.label2.setStyleSheet('font-size: 32px; color: black; font-weight: 900')

        self.label3 = QLabel('POP Music',self) #create label
        self.label3.setGeometry(QtCore.QRect(785,70,200,50)) #(水平位置,垂直位置,寬,高)
        self.label3.setStyleSheet('font-size: 32px; color: black; font-weight: 900')

        self.button2 = QPushButton('Module',self)
        self.button2.setGeometry(QtCore.QRect(450, 440, 200, 50))
        self.button2.setStyleSheet('background-color: 	#D0D0D0	; font-size: 24px; color: #EA7500; font-weight: 600; text-decoration: underline;')
        self.button2.clicked.connect(self.module)

        self.clicked = False

        self.checkbox1 = QCheckBox("Lo-Fi soothing", self)
        self.checkbox1.clicked.connect(self.checkboxClicked)
        self.checkbox1.setGeometry(QtCore.QRect(100,200,200,50)) #(水平位置,垂直位置,寬,高)
        self.checkbox1.setStyleSheet('font-size: 24px; color: black; font-weight: 400')
        
        self.checkbox2 = QCheckBox("Lo-Fi gentle", self)
        self.checkbox2.clicked.connect(self.checkboxClicked)
        self.checkbox2.setGeometry(QtCore.QRect(100,320,200,50)) #(水平位置,垂直位置,寬,高)
        self.checkbox2.setStyleSheet('font-size: 24px; color: black; font-weight: 400')

        self.checkbox3 = QCheckBox("Schubert", self)
        self.checkbox3.clicked.connect(self.checkboxClicked)
        self.checkbox3.setGeometry(QtCore.QRect(485,200,200,50)) #(水平位置,垂直位置,寬,高)
        self.checkbox3.setStyleSheet('font-size: 24px; color: black; font-weight: 400')

        self.checkbox4 = QCheckBox("Beethoven", self)
        self.checkbox4.clicked.connect(self.checkboxClicked)
        self.checkbox4.setGeometry(QtCore.QRect(485,320,200,50)) #(水平位置,垂直位置,寬,高)
        self.checkbox4.setStyleSheet('font-size: 24px; color: black; font-weight: 400')

        self.checkbox5 = QCheckBox("Chainsmokers", self)
        self.checkbox5.clicked.connect(self.checkboxClicked)
        self.checkbox5.setGeometry(QtCore.QRect(785,200,200,50)) #(水平位置,垂直位置,寬,高)
        self.checkbox5.setStyleSheet('font-size: 24px; color: black; font-weight: 400')

        self.checkbox6 = QCheckBox("Ariana Grande", self)
        self.checkbox6.clicked.connect(self.checkboxClicked)
        self.checkbox6.setGeometry(QtCore.QRect(785,320,200,50)) #(水平位置,垂直位置,寬,高)
        self.checkbox6.setStyleSheet('font-size: 24px; color: black; font-weight: 400')

    def checkboxClicked(self):
        if not self.clicked:
            # 传递变量的操作
            self.clicked = True
        else:
            self.clicked = False

    
    def goToPage2(self):
        # 觸發轉移到Page2
        main_widget.setCurrentIndex(1)  # 將索引設置為1，即第二個頁面

    def module(self):
        notes = []
        folders = []  # List of folder paths

        if self.checkbox1.isChecked():
            folders.append("Lo-Fi Hip Hop/1")
        if self.checkbox2.isChecked():
            folders.append("Lo-Fi Hip Hop/2")
        if self.checkbox3.isChecked():
            folders.append("Classical/Schubert")
        if self.checkbox4.isChecked():
            folders.append("Classical/Beethoven")
        if self.checkbox5.isChecked():
            folders.append("pop/1")
        if self.checkbox6.isChecked():
            folders.append("pop/2")

        for folder in folders:
            file_pattern = folder + "/*.mid"  # Pattern to match MIDI files
            files = glob.glob(file_pattern)

            for file in files:
                midi = converter.parse(file)
                notes_to_parse = None
                parts = instrument.partitionByInstrument(midi)
                if parts: # file has instrument parts
                    notes_to_parse = parts.parts[0].recurse()
                else: # file has notes in a flat structure
                    notes_to_parse = midi.flat.notes
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
        sequence_length = 90
        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        n_vocab = len(pitchnames)
        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab)
        network_output = np_utils.to_categorical(network_output)

        model = Sequential()
        model.add(LSTM(
            256,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128))
        model.add(Dense(128))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        filepath = "otherweight/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"    
        checkpoint = ModelCheckpoint(
            filepath, monitor='loss', 
            verbose=0,        
            save_best_only=True,        
            mode='min'
        )    
        callbacks_list = [checkpoint]     
        model.fit(network_input, network_output, epochs=70, batch_size=64, callbacks=callbacks_list)
        
        # 觸發轉移到Page1
        main_widget.setCurrentIndex(0)  # 將索引設置為0，即第一個頁面

class Page6(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.original_pixmap1 = QPixmap('images/review.jpg')

        # 設定目標寬度和高度
        target_width = 1080
        target_height = 608
        
        self.pixmap1 = self.original_pixmap1.scaled(target_width, target_height)

        self.image_label1 = QLabel(self)
        self.image_label1.setPixmap(self.pixmap1)
        self.image_label1.setGeometry(0, 0, 1080, 608)

        self.background1 = QWidget(self) #background
        self.background1.setGeometry(470,528,150,45) #(水平位置,垂直位置,寬,高)
        self.background1.setStyleSheet("background-color: #FFDCB9; border-radius: 20px;")

        self.background2 = QWidget(self) #background
        self.background2.setGeometry(60,502,120,40) #(水平位置,垂直位置,寬,高)
        self.background2.setStyleSheet("background-color: #7B7B7B; border-radius: 20px;")


        self.button1 = QPushButton('GO BACK ',self)
        self.button1.setGeometry(QtCore.QRect(450, 525, 200, 50))
        self.button1.setStyleSheet('background-color: transparent; font-size: 22px; color: #003D79; font-weight: 900')
        self.button1.clicked.connect(self.goToPage1)
        
        self.button2 = QPushButton('Load File', self)
        self.button2.setGeometry(QtCore.QRect(60, 470, 120, 100))
        self.button2.setStyleSheet('background-color: transparent; font-size: 22px; color: #FFFF37; font-weight: 900')
        self.button2.clicked.connect(self.loadFile)

        self.label = QLabel('No file loaded',self) #create label
        self.label.setGeometry(QtCore.QRect(60,536,300,50)) #(水平位置,垂直位置,寬,高)
        self.label.setStyleSheet('font-size: 18px; color: white; font-weight: 600')

        self.button3 = QPushButton('Play', self)
        self.button3.setGeometry(QtCore.QRect(490, 100, 120, 200))
        self.button3.setStyleSheet('background-color: transparent; font-size: 40px; color: #FFFF37; font-weight: 900')
        self.button3.clicked.connect(self.play)
        self.button3.hide()

        self.startTextAnimation()

    def startTextAnimation(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateButtonStyle)
        self.timer.start(750)  # Update the style every 7000 milliseconds (0.7 second)

    def updateButtonStyle(self):
        current_color = self.button3.palette().color(self.button3.foregroundRole())
        if current_color == QColor("#FFFF37"):
            self.button3.setStyleSheet('background-color: rgba(0, 0, 0, 0); font-size: 24px; color: transparent; font-weight: 900')

        else:
            self.button3.setStyleSheet('background-color: rgba(0, 0, 0, 0); font-size: 24px; color: #FFFF37; font-weight: 900')

    def loadFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, caption='Upload MIDI File', filter="MIDI Files (*.mid *.midi)")
        self.midi_file_path = filename
        self.label.setText(os.path.basename(filename))
        if filename:
            self.button3.show()

    def play(self):
        pygame.init()

        # 設定音訊參數
        sample_rate = 44100
        bit_depth = -16
        channels = 2

        # 設定音訊輸出
        pygame.mixer.init(sample_rate, bit_depth, channels)

        # 播放MIDI檔案
        pygame.mixer.music.load(self.midi_file_path)
        pygame.mixer.music.play()

        # 等待音樂播放結束
        while pygame.mixer.music.get_busy():
            pass


    def goToPage1(self):
        # 觸發轉移到Page1
        main_widget.setCurrentIndex(0)  # 將索引設置為0，即第一個頁面

app = QApplication([sys.argv])
app.setApplicationName("music generation")
main_widget = QStackedWidget()
main_widget.setGeometry(300, 100, 1080, 608)
page1 = Page1()
page2 = Page2()
page3_1 = Page3_1()
page3_2 = Page3_2()
page3_3 = Page3_3()
page4 = Page4()
page5 = Page5()
page6 = Page6()
main_widget.addWidget(page1)
main_widget.addWidget(page2)
main_widget.addWidget(page3_1)
main_widget.addWidget(page3_2)
main_widget.addWidget(page3_3)
main_widget.addWidget(page4)
main_widget.addWidget(page5)
main_widget.addWidget(page6)
main_widget.show()
sys.exit(app.exec_())
