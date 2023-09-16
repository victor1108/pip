import sys, random, mysql.connector, subprocess, pyaudio, wave, openai, shutil, os, threading, re
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt,QUrl, QTimer
from PyQt5.QtGui import QPainter, QMovie, QCursor, QPixmap
from PyQt5.QtWidgets import  QDesktopWidget, QMessageBox, QTableView, QFileDialog
from functools import partial
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QItemDelegate
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
from pydub import AudioSegment
from datetime import datetime
import speech_recognition as sr
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pydub.utils import make_chunks
from openai.error import RateLimitError

class GifWindow1(QtWidgets.QWidget):
    def __init__(self, gif_path):
        super().__init__()
        self.gif_path = gif_path
        self.initUI()

    def initUI(self):
        self.setWindowTitle('GIF Window')
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.resize(59, 57)  # 設置視窗大小

        frame = QtWidgets.QFrame(self)

        frame.setGeometry(2, 2, self.width(), self.height())

        self.label = QtWidgets.QLabel(frame)
        self.label.setGeometry(0, 0, frame.width(), frame.height())
        movie = QtGui.QMovie(self.gif_path)
        movie.setScaledSize(QtCore.QSize(frame.width(), frame.height()))
        self.label.setMovie(movie)
        movie.start()

        # 設置計時器，在一段時間後關閉視窗
        self.timer = QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.closeWindow)
        self.timer.start(10000)  # 隨機時間後關閉視窗

        self.setFixedSize(self.size())
        self.move(1224+300,460+60)
        self.show()


    def closeWindow(self):
        self.close()

class GifWindow(QtWidgets.QWidget):
    def __init__(self, gif_path):
        super().__init__()
        self.gif_path = gif_path
        self.initUI()

    def initUI(self):
        self.setWindowTitle('GIF Window')
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.resize(576, 432)  # 設置視窗大小

        frame = QtWidgets.QFrame(self)
        frame.setStyleSheet("border: 2px solid black; border-radius: 5px;")
        frame.setGeometry(2, 2, self.width() - 4, self.height() - 4)

        self.label = QtWidgets.QLabel(frame)
        self.label.setGeometry(0, 0, frame.width(), frame.height())
        movie = QtGui.QMovie(self.gif_path)
        movie.setScaledSize(QtCore.QSize(frame.width(), frame.height()))
        self.label.setMovie(movie)
        movie.start()

        # 設置計時器，在一段時間後關閉視窗
        self.timer = QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.closeWindow)
        timeout = random.randrange(3000, 3500)  # 生成3至5秒之間的隨機時間
        self.timer.start(timeout)  # 隨機時間後關閉視窗

        self.centerOnScreen()
        self.setFixedSize(self.size())

        self.show()

    def centerOnScreen(self):
        screen_geometry = QtWidgets.QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def closeWindow(self):
        self.close()

class AnimatedButton(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(0, 0, 155, 207))
        self.movie = QMovie("pic/images.gif")
        self.movie.setScaledSize(QtCore.QSize(155, 207))
        self.label.setMovie(self.movie)
        self.movie.start()
        self.movie.stop()

    def enterEvent(self, event):
        self.movie.start()

    def leaveEvent(self, event):
        self.movie.stop()
        self.movie.jumpToFrame(0)

class AnimatedButton1(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(0, 0, 113, 150))
        self.movie = QMovie("pic/images2-unscreen.gif")
        self.movie.setScaledSize(QtCore.QSize(113, 150))
        self.label.setMovie(self.movie)

    def enterEvent(self, event):
        self.movie.start()
        

    def leaveEvent(self, event):
        self.movie.stop()
        self.movie.jumpToFrame(0)
        self.hide()

class AnimatedButton2(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(0, 0, 113, 150))
        self.movie = QMovie("pic/images2-unscreen.gif")
        self.movie.setScaledSize(QtCore.QSize(113, 150))
        self.label.setMovie(self.movie)

    def enterEvent(self, event):
        self.movie.start()
        
    def leaveEvent(self, event):
        self.movie.stop()
        self.movie.jumpToFrame(0)
        self.hide()

class mainButton1(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)

    def enterEvent(self, event):
        self.parent().findChild(AnimatedButton1, "anibutton").show()

class mainButton2(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)

    def enterEvent(self, event):
        self.parent().findChild(AnimatedButton1, "anibutton1").show()

class mainButton3(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)

    def enterEvent(self, event):
        self.parent().findChild(AnimatedButton2, "anibutton2").show()

def apply_bandpass_filter(data, sample_rate, low_freq, high_freq):
    nyquist_freq = 0.5 * sample_rate
    low_cutoff = low_freq / nyquist_freq
    high_cutoff = high_freq / nyquist_freq

    filter_order = 100
    filter_coeffs = firwin(filter_order, [low_cutoff, high_cutoff], pass_zero=False)

    filtered_data = lfilter(filter_coeffs, 1.0, data)

    return filtered_data

def rms_to_db(rms, reference=5e-2):
    return 20 * np.log10(rms / reference)

def measure_volume(file_path):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    audio = pyaudio.PyAudio()

    wf = wave.open(file_path, 'rb')
    sample_width = wf.getsampwidth()
    data = wf.readframes(CHUNK)

    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlabel('時間（秒）')
    ax.set_ylabel('音量（dB）')
    ax.set_title('音量圖表')

    volume_data = []

    total_duration = 0
    current_duration = 0

    while len(data) > 0:
        audio_data = np.frombuffer(data, dtype=np.int16)

        filtered_data = apply_bandpass_filter(audio_data, RATE, 1000, 8000)

        volume = rms_to_db(np.sqrt(np.mean(filtered_data ** 2)))

        if volume >= 15:
            volume_data.append(volume)
            current_duration += CHUNK / (RATE * sample_width)
        else:
            if len(volume_data) > 0:
                total_duration += current_duration
                current_duration = 0

            volume = np.nan
        time_axis = np.arange(len(volume_data)) * CHUNK / RATE

        line.set_data(time_axis, volume_data)
        data = wf.readframes(CHUNK)

    total_duration += current_duration

    audio.terminate()
    wf.close()

    if total_duration > 0:
        average_volume = np.mean(volume_data)
        print("Average Volume (dB):", average_volume)
        print("Total Duration:", total_duration, "seconds")
    else:
        print("No speech detected.")
    return volume_data

class UploadWindow(QtWidgets.QDialog):
    def __init__(self, parent=None,account=None):
        super().__init__(parent)
        self.setWindowTitle("Upload File")
        self.setMinimumSize(300, 100)   
        self.account = account
        layout = QtWidgets.QVBoxLayout()
        self.file_label = QtWidgets.QLabel()
        layout.addWidget(self.file_label)
        self.setLayout(layout)
        self.transcription_display = QtWidgets.QTextEdit()
        layout.addWidget(self.transcription_display)
        self.transcription_display.hide() 
        self.file_path = None
        browse_button = QtWidgets.QPushButton("Browse")
        layout.addWidget(browse_button)
        browse_button.clicked.connect(self.browse_file)

        upload_button = QtWidgets.QPushButton("Upload")
        layout.addWidget(upload_button)
        upload_button.clicked.connect(self.upload_file)

    def browse_file(self):
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(self, "選擇音檔", "", "音檔 (*.wav)")
        self.file_label.setText(self.file_path)


    def upload_file(self):
        self.file_label.setText("")
        if self.file_path:
            self.accept()

            # 創建等待窗口
            waiting_window1 = WaitingWindow1(parent=self)
            waiting_window1.show()

            # 執行識別和評分的
            volume_data = measure_volume(self.file_path)  # 獲取音量數據
            text = self.transcribe(self.file_path)
            scores, total_score, average_score = self.evaluate_speech(text)
            speeds, total_average_speed = self.perform_speed_analysis(self.file_path)
            average_volume = np.mean(volume_data)
            
            # 關閉等待
            waiting_window1.close()

            # 顯示結果
            result_window = ResultWindow(scores, total_score, average_score, speeds, total_average_speed, volume_data, self)
            result_window.exec_()
            self.transcription_display.setText(text)

            # 將上傳的檔案資訊存入資料庫
            account = self.account
            try:
                # 連接資料庫，執行插入操作
                self.db = mysql.connector.connect(
                    host='localhost',
                    user='root',
                    password='daidai',
                    database='123'
                )
                cursor = self.db.cursor()
                cursor.execute('SELECT MAX(gradeid) FROM grade')
                result = cursor.fetchone()
                if result[0] is None:
                    gradeid = 1
                else:
                    gradeid = result[0] + 1

                upload_time = datetime.now()
                cursor = self.db.cursor()
                scores_string = ", ".join([f"{criterion}: {score}" for criterion, score in scores.items()])

                query = 'INSERT INTO grade (gradeid, account, speech, grade, time, speed) VALUES (%s, %s, %s, %s, %s, %s)'
                values = (gradeid, account, self.file_path, scores_string, upload_time, f"總平均語速為 {total_average_speed:.2f}字/秒，平均音量為 {np.mean(volume_data):.2f} dB")
                cursor.execute(query, values)
                self.db.commit()

            except mysql.connector.Error as err:
                QtWidgets.QMessageBox.warning(self, '警告', str(err))


    def transcribe(self, file_path):
        try:
            openai.api_key = '0sk-tifdeUVE5oeV8RpW146NT3BlbkFJeddkwiIzdj4MprPeeQw0'
            audio_file = open(file_path, "rb")
            transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file, response_format="text")
            text = transcript.replace(" ", "")
            self.transcription_display.setText(text)

            # 讀取音頻檔案，進行降噪處理
            sound = AudioSegment.from_file(file_path, format="wav")
            # 使用低通濾波器降噪
            sound = sound.low_pass_filter(3000)
            # 使用高通濾波器降噪
            sound = sound.high_pass_filter(200)
            # 正規化音量
            sound = sound.normalize()
            # 將處理後的音頻存儲為wav格式檔案
            sound.export(file_path, format="wav")

            return text
        except RateLimitError as e:
            print(e)
    
    def evaluate_speech(self, speech):
        try:
            evaluation_criteria = [
                "內容完整性",
                "語言表達",
                "邏輯結構",
                "語法正確性",
                "表達力",
            ]

            scores = {
                "內容完整性": 0,
                "語言表達": 0,
                "邏輯結構": 0,
                "語法正確性":0,
                "表達力":0,
            }
            for criterion in evaluation_criteria:
                prompt = f"請您扮演一位老師，以同樣的標準幫我給以下講稿做評分：\n\n{speech}\n\n請對「{criterion}」進行評分，滿分為10分。"
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=100,
                    n=1,
                    stop=None,
                    temperature=0.8,
                )
                score_text = response.choices[0].text.strip()
                try:
                    score = int(re.search(r'\d+', score_text).group())
                    scores[criterion] = score
                    print(f"{criterion}：{score}分")
                except AttributeError:
                    print(f"未能獲取到 '{criterion}' 的分數。")

            total_score = sum(scores.values())
            if len(scores) > 0:
                average_score = total_score / len(scores)
            else:
                average_score = 0
            print(f"總分：{total_score}分")
            print(f"平均分：{average_score:.2f}分")

            return scores, total_score, average_score
        except RateLimitError as e:
            QtWidgets.QMessageBox.warning(self, "Warning", str(e))
            return
    
    def perform_speed_analysis(self, file_path):
        list_duration=[]
        list_duration2=[]
        Speech=[]
        Speed=[]
        if not os.path.exists('temdir2'):
            os.mkdir('temdir2')

        audiofile = AudioSegment.from_file(file_path, "wav")

        chunklist=make_chunks(audiofile, 10000)
        for i, chunk in enumerate(chunklist):
            chunk_name="temdir2/chunk{0}.wav".format(i)
            print("存檔:",chunk_name)
            chunk.export(chunk_name,format="wav")
            song = AudioSegment.from_mp3("{}".format(chunk_name)) 
            duration = song.duration_seconds
            list_duration.append(duration)
        summ=0    
        for i in range(len(list_duration)):
            summ+=list_duration[i]
            list_duration2.append(summ)
        r=sr.Recognizer()
        print("開始翻譯...")
        file=open("phthon_sr.txt","w")
        for i in range(len(chunklist)):
            try:
                with sr.WavFile("temdir2/chunk{}.wav".format(i)) as source:
                    audio=r.record(source)
                result=r.recognize_google(audio,language="zh-TW")
                print("  "+str(result))
                Speech.append(result)
                file.write(result)
            except sr.UnknownValueError:
                print("Google Speech Recognition 無法辨識此語音!")
            except sr.RequestError as e:
                print("無法由 Google Speech Recognition 取得結果; {0}".format(e))
        file.close()
        print("翻譯結束!")
        shutil.rmtree('temdir2')
        print(Speech)
        print(list_duration)
        print(list_duration2)  
        print(len(list_duration))

        start=0
        avgspeed=0
        totalword=0
        speeds = []
        for i in range(len(Speech)):
            avgspeed = (len(Speech[i])) / (list_duration[i])
            totalword += len(Speech[i])
            Speed.append(avgspeed)
            print(str(start) + '~' + str(start + list_duration[i]) + '秒的平均語速是' + str(avgspeed) + '字/秒')
            speeds.append((start, start + list_duration[i], avgspeed))
            start += 10
        avgspeed = (totalword) / (list_duration2[-1])
        print("0~" + str(list_duration2[-1]) + "秒的總平均語速是" + str(avgspeed) + '字/秒')
        print(Speed)
        return speeds, avgspeed

class WaitingWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("錄製")
        self.setFixedSize(200, 100)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        label = QtWidgets.QLabel("按下結束錄製以開始辨識")
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)

class WaitingWindow1(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("辨識中")
        self.setFixedSize(200, 100)

        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)

        label = QtWidgets.QLabel("辨識中請稍後")
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)

class ResultWindow(QtWidgets.QDialog):
    def __init__(self, scores, total_score, average_score, speeds, total_average_speed, volume_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Speech Evaluation Result")
        self.setMinimumSize(400, 200)
        self.scores = scores  # 添加 scores 屬性
        self.total_score = total_score
        self.average_score = average_score
        self.speeds = speeds
        self.total_average_speed = total_average_speed
        self.volume_data = volume_data
        self.volume_data = volume_data

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        evaluation_criteria = [
            "內容完整性",
            "語言表達",
            "邏輯結構",
            "語法正確性",
            "表達力",
        ]

        for criterion, score in scores.items():
            label = QtWidgets.QLabel(f"{criterion}: {score}分")
            layout.addWidget(label)

        speeds_label1 = QtWidgets.QLabel("語速結果：")
        layout.addWidget(speeds_label1)

        speeds_list1 = QtWidgets.QListWidget()
        layout.addWidget(speeds_list1)
        self.transcription_display = QtWidgets.QTextEdit(self)
        layout.addWidget(self.transcription_display)


        for speed in speeds:
            speeds_list1.addItem(f"{speed[0]}~{speed[1]}秒的平均語速：{speed[2]:.2f}字/秒")

        total_average_label = QtWidgets.QLabel(f"總平均語速：{total_average_speed:.2f}字/秒")
        layout.addWidget(total_average_label)

        average_volume = np.mean(volume_data)
        average_volume_label = QtWidgets.QLabel(f"平均音量：{average_volume:.2f} dB")
        layout.addWidget(average_volume_label)

        # 繪製音量圖表
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.plot_volume(self.volume_data)
        suggestions = self.provide_suggestions(volume_data, total_average_speed)
        suggestion_text = "\n".join(suggestions)
        self.transcription_display.append("建議：\n" + suggestion_text)
        self.transcription_display.setReadOnly(True)
    def closeEvent(self,event):
        event.accept()
    def plot_volume(self, volume_data):
        sample_rate = 44100  # 根據音頻文件的採樣率進行調整
        audio_length = len(volume_data) / sample_rate
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        # 生成時間軸
        time_axis = np.arange(len(volume_data)) * CHUNK / RATE

        ax = self.figure.add_subplot(111)
        ax.plot(time_axis, volume_data)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Volume (dB)')
        ax.set_title('Volume Chart')
        ax.grid(True)

        self.canvas.draw()
    
    def provide_suggestions(self, volume_data, total_average_speed):
        average_volume = np.mean(volume_data)

        suggestions = []

        if average_volume > 70:
            suggestions.append("音量偏高，建議遠離麥克風或降低麥克風增益。")
        elif 60 <= average_volume <= 70:
            suggestions.append("音量稍高，可以考慮遠離麥克風或降低麥克風增益。")
        elif 50 <= average_volume < 60:
            suggestions.append("音量偏低，建議靠近麥克風或增加麥克風增益。")
        else:
            suggestions.append("音量適中，音量調整良好。")

        if total_average_speed > 2.5:
            suggestions.append("語速較快，建議放慢語速以提高清晰度。")
        elif 2 <= total_average_speed <= 2.5:
            suggestions.append("語速稍快，可以考慮放慢語速以提高清晰度。")
        elif 1.5 <= total_average_speed < 2:
            suggestions.append("語速稍慢，建議增加語速以提高流暢度。")
        else:
            suggestions.append("語速適中，語速調整良好。")

        if 0 <= self.scores["內容完整性"] < 5:
            suggestions.append("內容完整性較低，需要加強表達全面、具體、清晰。")
        elif 5 <= self.scores["內容完整性"] < 7:
            suggestions.append("內容完整性有待提高，請確保表達全面、具體、清晰。")
        elif 7 <= self.scores["內容完整性"] < 9:
            suggestions.append("內容完整性良好，可以繼續保持這個水準。")
        else:
            suggestions.append("內容完整性很好，保持這個水準。")

        if 0 <= self.scores["語言表達"] < 5:
            suggestions.append("語言表達需要大幅提高，請注意詞彙選擇、句法結構和語法正確性。")
        elif 5 <= self.scores["語言表達"] < 7:
            suggestions.append("語言表達有待提高，請注意詞彙選擇、句法結構和語法正確性。")
        elif 7 <= self.scores["語言表達"] < 9:
            suggestions.append("語言表達很出色，可以繼續保持這個水準。")
        else:
            suggestions.append("語言表達非常出色，保持這個水準。")

        if 0 <= self.scores["邏輯結構"] < 5:
            suggestions.append("邏輯結構需要大幅提高，請確保思路清晰、結構合理、論點有力。")
        elif 5 <= self.scores["邏輯結構"] < 7:
            suggestions.append("邏輯結構有待提高，請確保思路清晰、結構合理、論點有力。")
        elif 7 <= self.scores["邏輯結構"] < 9:
            suggestions.append("邏輯結構很優秀，可以繼續保持這個水準。")
        else:
            suggestions.append("邏輯結構非常優秀，保持這個水準。")

        if 0 <= self.scores["語法正確性"] < 5:
            suggestions.append("語法正確性需要大幅提高，請注意語法錯誤和句子結構。")
        elif 5 <= self.scores["語法正確性"] < 7:
            suggestions.append("語法正確性有待提高，請注意語法錯誤和句子結構。")
        elif 7 <= self.scores["語法正確性"] < 9:
            suggestions.append("語法正確性很好，可以繼續保持這個水準。")
        else:
            suggestions.append("語法正確性非常好，保持這個水準。")

        if 0 <= self.scores["表達力"] < 5:
            suggestions.append("表達力需要大幅提高，請注重情感表達、語調變化和節奏感。")
        elif 5 <= self.scores["表達力"] < 7:
            suggestions.append("表達力有待提高，請注重情感表達、語調變化和節奏感。")
        elif 7 <= self.scores["表達力"] < 9:
            suggestions.append("表達力很出色，可以繼續保持這個水準。")
        else:
            suggestions.append("表達力非常出色，保持這個水準。")

        return suggestions

class AddVideoDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        account = main_window.lineEdit.text()
        self.setWindowTitle("Add Video")
        self.setMinimumSize(400, 200)

        self.url_label = QtWidgets.QLabel("Video URL:")
        self.url_input = QtWidgets.QLineEdit()
        self.title_label = QtWidgets.QLabel("Video Title:")
        self.title_input = QtWidgets.QLineEdit()
        self.upload_button = QtWidgets.QPushButton("Upload")
        self.upload_button.clicked.connect(self.upload_video)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.url_label)
        layout.addWidget(self.url_input)
        layout.addWidget(self.title_label)
        layout.addWidget(self.title_input)
        layout.addWidget(self.upload_button)

        self.setLayout(layout)

    def upload_video(self):
        print(123)
        url = self.url_input.text()
        title = self.title_input.text()
        
        if not url or not title:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please enter both URL and Title.")
            return
        account = main_window.lineEdit.text()
        try:
            self.db = mysql.connector.connect(
                host='localhost',
                user='root',
                password='daidai',
                database='123'
            )
            cursor = self.db.cursor()
            cursor.execute('SELECT MAX(videoid) FROM video')
            result = cursor.fetchone()
            if result[0] is None:
                videoid = 1
            else:
                videoid = result[0] + 1
            cursor = self.db.cursor()
            query = "INSERT INTO video (account, url, title,videoid) VALUES (%s, %s, %s,%s)"
            values = (main_window.lineEdit.text(), url, title,videoid)
            cursor.execute(query, values)
            self.db.commit()

            QtWidgets.QMessageBox.information(self, "Success", "Video uploaded successfully.")
            self.accept()

        except mysql.connector.Error as err:
            QtWidgets.QMessageBox.warning(self, "Warning", str(err))

class YouTubePlayer(QWidget):

    def __init__(self, video_id, parent=None):
        super().__init__(parent)
        self.video_id = video_id
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        topLayout = QHBoxLayout()
        self.layout.addLayout(topLayout)

        self.addWebView(f'https://www.youtube.com/watch?v={self.video_id}')

        self.mediaPlayer = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self.mediaPlayer.mediaStatusChanged.connect(self.handleMediaStatusChanged)

    def addWebView(self, video_url):
        settings = QWebEngineSettings.defaultSettings()
        settings.setAttribute(QWebEngineSettings.PluginsEnabled, True)
        self.webview = QWebEngineView()
        self.webview.setUrl(QUrl(f'https://www.youtube.com/embed/{self.video_id}?rel=0'))

        self.layout.addWidget(self.webview)


    def handleMediaStatusChanged(self, status):
        if status == QMediaPlayer.EndOfMedia:
            # 當影片撥放到結尾時停止播放
            self.mediaPlayer.stop()

    def closeEvent(self, event):
        self.webview.stop()
        self.webview.setZoomFactor(0.0)  
        self.webview.close()
        self.webview.setParent(None)
        event.accept()

class YouTubeWindow(QWidget):

    def __init__(self, video_id):
        super().__init__()
        self.setWindowTitle('YouTube Video Player')
        self.setMinimumSize(800, 600)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.player = YouTubePlayer(video_id, parent=self)
        self.layout.addWidget(self.player)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_video_id = None
        self.selected_video = None
        self.recording = False
        self.stream = None
        self.p = pyaudio.PyAudio()
        self.pixmap = QtGui.QPixmap('pic/homepage.png')
        self.initUI()
        self.centerOnScreen()
        self.setFixedSize(self.size())
        self.setWindowIcon(QtGui.QIcon("pic/icon.png"))
                # 設置滑鼠造型動畫
        cursor_animation_paths = ['pic/11.png',
                                  'pic/22.png',
                                  'pic/33.png',
                                  'pic/44.png',
                                  ]

        
        self.cursor_frames = [QPixmap(path) for path in cursor_animation_paths]
        self.current_frame = 0

        # 設置滑鼠造型
        cursor_pixmap = self.cursor_frames[self.current_frame]
        cursor = QCursor(cursor_pixmap)
        self.setCursor(cursor)

        # 設置更新滑鼠造型的計時器
        self.timer1 = QTimer()
        self.timer1.timeout.connect(self.updateCursor)
        self.timer1.start(130)  # 每130毫秒更新一次滑鼠造型


    def showGifWindow(self):
        gif_path = "pic/loading.gif"  # 替換為你想要顯示的 GIF 檔案的路徑
        gif_window = GifWindow(gif_path)
        gif_window.centerOnScreen()  # 將此行添加到顯示 GIF 視窗的方法中
        self.gif_window = gif_window  # 將 gif_window 設置為成員變數
        gif_window.show()

    def showGifWindow1(self):
        gif_path = "pic/images3.gif"  # 替換為你想要顯示的 GIF 檔案的路徑
        gif_window = GifWindow1(gif_path)
        self.gif_window = gif_window  # 將 gif_window 設置為成員變數
        gif_window.show()
    
    def updateCursor(self):
        # 更新滑鼠造型
        self.current_frame = (self.current_frame + 1) % len(self.cursor_frames)
        cursor_pixmap = self.cursor_frames[self.current_frame]
        cursor = QCursor(cursor_pixmap)
        self.setCursor(cursor)

    

    def paintEvent(self, event=None):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)

    def centerOnScreen(self):
        # 取得視窗
        qr = self.frameGeometry()
        # 取得螢幕中心點
        cp = QDesktopWidget().availableGeometry().center()
        # 將視窗的中心點移到螢幕的中心點
        qr.moveCenter(cp)
        # 將視窗的左上角移動到其框架的左上角，即讓視窗出現在螢幕的中心
        self.move(qr.topLeft())

    def initUI(self):
        self.setObjectName("MainWindow")
        self.resize(1280, 960)
        self.setWindowTitle('PIP')
        icon = QtGui.QIcon.fromTheme("PIP")
        self.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(515, 420, 250, 137))
        self.pushButton.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setStyleSheet("""
            QPushButton {
                background-image: url(pic/login.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/loginhover.png);
            }
            """)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(515, 560, 250, 137))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setStyleSheet("""
            QPushButton {
                background-image: url(pic/homepageregister.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/homepageregisterhover.png);
            }
            """)
        
        self.pushButton.show()
        self.pushButton_2.show()
        self.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)


        QtCore.QMetaObject.connectSlotsByName(self)

        self.pushButton.clicked.connect(self.gologin)
        self.pushButton_2.clicked.connect(self.goregis)



        #login
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(525, 455, 290, 40))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setStyleSheet("""
            QLineEdit {
                font-family: "微軟正黑體";
                font-size: 20pt;
                padding: 2px;
                border-radius: 3px;
                background-color: #f2d599;
            }
            """)

        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(525, 555, 290, 40))
        self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_2.setStyleSheet("""
            QLineEdit {
                font-family: "微軟正黑體";
                font-size: 20pt;
                padding: 2px;
                border-radius: 3px;
                background-color: #f2d599;
            }
            """)

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(670, 650, 164, 104))
        self.pushButton_3.setObjectName("pushButton")
        self.pushButton_3.setStyleSheet("""
            QPushButton {
                background-image: url(pic/loginlogin.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/loginloginhover.png);
            }
            """)

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(455, 650, 164, 104))
        self.pushButton_4.setObjectName("pushButton_2")
        self.pushButton_4.setStyleSheet("""
            QPushButton {
                background-image: url(pic/loginreturn.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/loginreturnhover.png);
            }
            """)
        
        self.pushButton_3.hide()
        self.pushButton_4.hide()
        self.lineEdit.hide()
        self.lineEdit_2.hide()

        QtCore.QMetaObject.connectSlotsByName(self)
        self.pushButton_3.clicked.connect(self.Login)
        self.pushButton_4.clicked.connect(self.loginback)



        #帳號
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(510, 510, 120, 40))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_3.setStyleSheet("""
            QLineEdit {
                font-family: "微軟正黑體";
                font-size: 20pt;
                padding: 2px;
                border-radius: 3px;
                background-color: #f2d599;
            }
            """)

        #密碼
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(735, 510, 120, 40))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_4.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_4.setStyleSheet("""
            QLineEdit {
                font-family: "微軟正黑體";
                font-size: 20pt;
                padding: 2px;
                border-radius: 3px;
                background-color: #f2d599;
            }
            """)

 
        #名稱   
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setGeometry(QtCore.QRect(510, 435, 120, 40))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.lineEdit_5.setStyleSheet("""
            QLineEdit {
                font-family: "微軟正黑體";
                font-size: 20pt;
                padding: 2px;
                border-radius: 3px;
                background-color: #f2d599;
            }
            """)

        #性別
        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_6.setGeometry(QtCore.QRect(735, 435, 120, 40))
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.lineEdit_6.setStyleSheet("""
            QLineEdit {
                font-family: "微軟正黑體";
                font-size: 20pt;
                padding: 2px;
                border-radius: 3px;
                background-color: #f2d599;
            }
            """)

  
        #信箱
        self.lineEdit_7 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_7.setGeometry(QtCore.QRect(510, 590, 345, 40))
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.lineEdit_7.setStyleSheet("""
            QLineEdit {
                font-family: "微軟正黑體";
                font-size: 20pt;
                padding: 2px;
                border-radius: 3px;
                background-color: #f2d599;
            }
            """)

        #註冊
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(685, 650, 140, 95))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.setStyleSheet("""
            QPushButton {
                background-image: url(pic/registerregister.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/registerregisterhover.png);
            }
            """)

        #返回
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(475, 650, 140, 95))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.setStyleSheet("""
            QPushButton {
                background-image: url(pic/registerreturn.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/registerreturnhover.png);
            }
            """)
        
        self.pushButton_5.hide()
        self.pushButton_6.hide()
        self.lineEdit_3.hide()
        self.lineEdit_4.hide()
        self.lineEdit_5.hide()
        self.lineEdit_6.hide()
        self.lineEdit_7.hide()

        self.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        QtCore.QMetaObject.connectSlotsByName(self)
        self.pushButton_5.clicked.connect(self.Register)
        self.pushButton_6.clicked.connect(self.regisback)



        #社群
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(140, 150, 458, 423))
        self.pushButton_7.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_7.setStyleSheet("""
            QPushButton {
                background-image: url(pic/social.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/socialhover.png);
            }
            """)

        #辨識
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setGeometry(QtCore.QRect(400, 380, 514, 498))
        self.pushButton_8.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_8.setStyleSheet("""
            QPushButton {
                background-image: url(pic/identify.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/identifyhover.png);
            }
            """)

        #vr
        self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_9.setGeometry(QtCore.QRect(700, 40, 524, 420))
        self.pushButton_9.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_9.setStyleSheet("""
            QPushButton {
                background-image: url(pic/vr.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/vrhover.png);
            }
            """)

        #返回
        self.pushButton_10 = AnimatedButton(self.centralwidget)
        self.pushButton_10.setGeometry(QtCore.QRect(1095, 710, 155, 207))
        self.pushButton_10.setObjectName("pushButton_10")

        
        #個人資料
        self.pushButton_11 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_11.setGeometry(QtCore.QRect(945, 765, 143, 142))
        self.pushButton_11.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_11.setStyleSheet("""
            QPushButton {
                background-image: url(pic/profile.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/profilehover.png);
            }
            """)

        self.pushButton_7.hide()
        self.pushButton_8.hide()
        self.pushButton_9.hide()
        self.pushButton_10.hide()
        self.pushButton_11.hide()

        self.pushButton_7.clicked.connect(self.social)
        self.pushButton_7.clicked.connect(self.load_videos)
        self.pushButton_8.clicked.connect(self.identify)
        self.pushButton_10.clicked.connect(self.menuback)
        package_name = 'com.pip.pipvr'
        self.pushButton_9.clicked.connect(self.showGifWindow1)
        self.pushButton_9.clicked.connect(partial(self.launch_apk, package_name))
        
        try:
            self.pushButton_11.clicked.connect(self.profile)
        except Exception as e:
            print("Error connecting clicked signal to goregis method:", str(e))



        #帳號
        self.lineEdit_8 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_8.setGeometry(QtCore.QRect(360, 385, 280, 50))
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.lineEdit_8.setReadOnly(True)
        self.lineEdit_8.setStyleSheet("""
            QLineEdit {
                font-family: "微軟正黑體";
                font-size: 20pt;
                padding: 5px;
                border-radius: 3px;
            }
            """)


        #密碼
        self.lineEdit_9 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_9.setGeometry(QtCore.QRect(830, 385, 280, 50))
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.lineEdit_9.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_9.setReadOnly(True)
        self.lineEdit_9.setStyleSheet("""
            QLineEdit {
                font-family: "微軟正黑體";
                font-size: 20pt;
                padding: 5px;
                border-radius: 3px;
            }
            """)
        #名稱   
        self.lineEdit_10 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_10.setGeometry(QtCore.QRect(360, 215, 280, 50))
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.lineEdit_10.setReadOnly(True)
        self.lineEdit_10.setStyleSheet("""
            QLineEdit {
                font-family: "微軟正黑體";
                font-size: 20pt;
                padding: 5px;
                border-radius: 3px;
            }
            """)

        #性別
        self.lineEdit_11 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_11.setGeometry(QtCore.QRect(830, 215, 280, 50))
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.lineEdit_11.setReadOnly(True)
        self.lineEdit_11.setStyleSheet("""
            QLineEdit {
                font-family: "微軟正黑體";
                font-size: 20pt;
                padding: 5px;
                border-radius: 3px;
            }
            """)
  
        #信箱
        self.lineEdit_12 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_12.setGeometry(QtCore.QRect(360, 555, 560, 50))
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.lineEdit_12.setReadOnly(True)
        self.lineEdit_12.setStyleSheet("""
            QLineEdit {
                font-family: "微軟正黑體";
                font-size: 20pt;
                padding: 5px;
                border-radius: 3px;
            }
            """)


        #返回
        self.pushButton_12 = mainButton3(self.centralwidget)
        self.pushButton_12.setGeometry(QtCore.QRect(1100, 780, 142, 143))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_12.setStyleSheet("""
            QPushButton {
                background-image: url(pic/return1.png); 
                border: none;
                outline: none;
            }
            """)
        
        self.anibutton2 = AnimatedButton2(self.centralwidget)
        self.anibutton2.setGeometry(QtCore.QRect(1115, 740, 113, 150))
        self.anibutton2.setObjectName("anibutton2")
        self.anibutton2.setStyleSheet("""
            QPushButton {
                 background-color: transparent;
                 outline: none;
            }
            """)
        self.anibutton2.hide()
        self.anibutton2.clicked.connect(self.userprofileback)
   

        self.pushButton_12.hide()
        self.lineEdit_8.hide()
        self.lineEdit_9.hide()
        self.lineEdit_10.hide()
        self.lineEdit_11.hide()
        self.lineEdit_12.hide()

        QtCore.QMetaObject.connectSlotsByName(self)
        self.pushButton_12.clicked.connect(self.userprofileback)

        

        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setObjectName("tableView")
        self.tableView.setGeometry(QtCore.QRect(340, 90, 600, 200))
        self.tableView.horizontalHeader().setStretchLastSection(True)
        self.tableView.verticalHeader().setVisible(False)
        self.tableView.setEditTriggers(QTableView.NoEditTriggers)
        
        self.tableView.setStyleSheet("""
            QTableView {
            color: black;
            gridline-color: black;
            border-color: rgb(242, 128, 133);
            font: 12px;
        }
        QHeaderView::section {
            background-color: rgb(71, 153, 176);
            color: white;
            height: 35px;
            font: 10px;
        }
        
        QScrollBar:vertical {
            background: rgb(188, 224, 235);
        }
        QScrollBar::handle:vertical {
            background: rgb(71, 153, 176);
        }
        QScrollBar:horizontal {
            background: rgb(188, 224, 235);
        }
        QScrollBar::handle:horizontal {
            background: rgb(71, 153, 176);
        }
        """)

        self.tableView_2 = QtWidgets.QTableView(self.centralwidget)
        self.tableView_2.setObjectName("tableView_2")
        self.tableView_2.setGeometry(QtCore.QRect(340, 330, 600, 200))
        self.tableView_2.horizontalHeader().setStretchLastSection(True)
        self.tableView_2.verticalHeader().setVisible(False)
        self.tableView_2.setEditTriggers(QTableView.NoEditTriggers)
        self.tableView_2.setStyleSheet("""
            QTableView {
            color: black;
            gridline-color: black;
            border-color: rgb(242, 128, 133);
            font: 12px;
        }
        QHeaderView::section {
            background-color: rgb(71, 153, 176);
            color: white;
            height: 35px;
            font: 10px;
        }
        
        QScrollBar:vertical {
            background: rgb(188, 224, 235);
        }
        QScrollBar::handle:vertical {
            background: rgb(71, 153, 176);
        }
        QScrollBar:horizontal {
            background: rgb(188, 224, 235);
        }
        QScrollBar::handle:horizontal {
            background: rgb(71, 153, 176);
        }
        """)
        
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setObjectName("textEdit")
        self.textEdit.setGeometry(QtCore.QRect(340, 580, 600, 200))
        self.textEdit.setStyleSheet("""
            QTextEdit {
                font-family: "微軟正黑體";
                font-size: 12pt;
                color: black;
                background-color: #F4F4F4;
                border: 2px solid #888888;
                border-radius: 5px;
                padding: 5px;
            }
        """)


        self.pushButton_13 = QtWidgets.QPushButton(self.centralwidget)   
        self.pushButton_13.setGeometry(QtCore.QRect(374, 830, 294, 89))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_13.setStyleSheet("""
            QPushButton {
                background-image: url(pic/play.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/playhover.png);
            }
            """)
        
        
        
        self.pushButton_14 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_14.setGeometry(QtCore.QRect(50, 830, 294, 89))
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_14.setStyleSheet("""
            QPushButton {
                background-image: url(pic/text.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/texthover.png);
            }
            """)




        self.pushButton_15 = mainButton2(self.centralwidget)
        self.pushButton_15.setGeometry(QtCore.QRect(1100, 780, 142, 143))
        self.pushButton_15.setObjectName("pushButton_15")
        self.pushButton_15.setStyleSheet("""
            QPushButton {
                background-image: url(pic/return1.png); 
                border: none;
                outline: none;
            }
            """)
        
        self.anibutton1 = AnimatedButton1(self.centralwidget)
        self.anibutton1.setGeometry(QtCore.QRect(1115, 740, 113, 150))
        self.anibutton1.setObjectName("anibutton1")
        self.anibutton1.setStyleSheet("""
            QPushButton {
                 background-color: transparent;
                 outline: none;
            }
            """)
        self.anibutton1.hide()
        self.anibutton1.clicked.connect(self.socialback)


        
        self.pushButton_16 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_16.setGeometry(QtCore.QRect(698, 830, 294, 89))
        self.pushButton_16.setObjectName("pushButton_16")
        self.pushButton_16.setStyleSheet("""
            QPushButton {
                background-image: url(pic/upload.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/uploadhover.png);
            }
            """)
        
        self.tableView.hide()
        self.tableView_2.hide()
        self.textEdit.hide()
        self.pushButton_13.hide()
        self.pushButton_14.hide()
        self.pushButton_15.hide()
        self.pushButton_16.hide()
        
        self.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)


        QtCore.QMetaObject.connectSlotsByName(self)
        self.load_videos()
        self.load_comments()
        self.tableView.clicked.connect(self.handle_video_clicked)
        self.pushButton_14.clicked.connect(self.submit_comment)
        self.pushButton_13.clicked.connect(self.handle_play)
        self.pushButton_15.clicked.connect(self.socialback)
        self.pushButton_16.clicked.connect(self.open_add_video_dialog)


        self.pushButton_17 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_17.setGeometry(QtCore.QRect(50, 830, 294, 89))
        self.pushButton_17.setObjectName("pushButton_17")
        self.pushButton_17.setStyleSheet("""
            QPushButton {
                background-image: url(pic/start.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/starthover.png);
            }
            """)


        self.pushButton_18 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_18.setGeometry(QtCore.QRect(698, 830, 294, 89))
        self.pushButton_18.setObjectName("pushButton_18")
        self.pushButton_18.setStyleSheet("""
            QPushButton {
                background-image: url(pic/end.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/endhover.png);
            }
            """)

        self.tableView_3 = QtWidgets.QTableView(self.centralwidget)
        self.tableView_3.setGeometry(QtCore.QRect(220, 150, 840, 500))
        self.tableView_3.setObjectName("tableView_3")
        self.tableView_3.horizontalHeader().setStretchLastSection(True)
        self.tableView_3.verticalHeader().setVisible(False)
        self.tableView_3.setEditTriggers(QTableView.NoEditTriggers)

        self.pushButton_19 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_19.setGeometry(QtCore.QRect(374, 700, 294, 89))
        self.pushButton_19.setObjectName("pushButton_19")
        self.pushButton_19.setStyleSheet("""
            QPushButton {
                background-image: url(pic/fileupload.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/fileuploadhover.png);
            }
            """)


        self.pushButton_20 = mainButton1(self.centralwidget)
        self.pushButton_20.setGeometry(QtCore.QRect(1100, 780, 142, 143))
        self.pushButton_20.setObjectName("pushButton_20")
        self.pushButton_20.setStyleSheet("""
            QPushButton {
                background-image: url(pic/return1.png); 
                border: none;
                outline: none;
            }
            """)
        
        self.anibutton = AnimatedButton1(self.centralwidget)
        self.anibutton.setGeometry(QtCore.QRect(1115, 740, 113, 150))
        self.anibutton.setObjectName("anibutton")
        self.anibutton.setStyleSheet("""
            QPushButton {
                 background-color: transparent;
                 outline: none;
            }
            """)
        self.anibutton.hide()
        self.anibutton.clicked.connect(self.textback)


        self.pushButton_21 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_21.setGeometry(QtCore.QRect(374, 830, 294, 89))
        self.pushButton_21.setObjectName("pushButton_21")
        self.pushButton_21.setStyleSheet("""
            QPushButton {
                background-image: url(pic/startplay.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/startplayhover.png);
            }
            """)
 
        self.pushButton_17.hide()
        self.pushButton_18.hide()
        self.pushButton_19.hide()
        self.pushButton_20.hide()
        self.pushButton_21.hide()
        self.tableView_3.hide()

        self.centralwidget.setObjectName("centralwidget")
        self.transcription_display = QtWidgets.QTextEdit(self.centralwidget)
        self.transcription_display.setGeometry(QtCore.QRect(2000, 2000, 211, 91))
        self.transcription_display.setObjectName("transcription_display")


        self.pushButton_22 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_22.setGeometry(QtCore.QRect(985, 104, 251, 76))
        self.pushButton_22.setObjectName("pushButton_22")
        self.pushButton_22.setStyleSheet("""
            QPushButton {
                background-image: url(pic/allvideo.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/allvideohover.png);
            }
            """)
        
        self.pushButton_23 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_23.setGeometry(QtCore.QRect(985, 200, 251, 76))
        self.pushButton_23.setObjectName("pushButton_23")
        self.pushButton_23.setStyleSheet("""
            QPushButton {
                background-image: url(pic/selfvideo.png); 
                border: none;
                outline: none;
            }
            QPushButton:hover {
                background-image: url(pic/selfvideohover.png);
            }
            """)
        
        self.pushButton_22.hide()
        self.pushButton_23.hide()

        self.pushButton_22.clicked.connect(self.load_videos)
        self.pushButton_23.clicked.connect(self.self_videos)
        self.pushButton_22.clicked.connect(self.load_comments)
        self.pushButton_23.clicked.connect(self.load_comments)
        self.pushButton_22.clicked.connect(self.handle_button_click)
        self.pushButton_23.clicked.connect(self.handle_button_click)


        # Add QTextEdit for displaying the transcription
        

        self.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        QtCore.QMetaObject.connectSlotsByName(self)

        self.pushButton_17.clicked.connect(self.start_recording)
        self.pushButton_18.clicked.connect(self.stop_recording_and_transcribe)
        self.pushButton_19.clicked.connect(self.open_upload_window)
        self.upload_window = None
        self.pushButton_20.clicked.connect(self.textback)
        self.pushButton_21.clicked.connect(self.play_audio)
        
    def gologin(self):
        self.pixmap = QtGui.QPixmap('pic/loginpage.png')
        self.update()
        self.pushButton.hide()
        self.pushButton_2.hide()
        self.pushButton_3.show()
        self.pushButton_4.show()
        self.lineEdit.show()
        self.lineEdit_2.show()

    def goregis(self):
        self.pixmap = QtGui.QPixmap('pic/register.png')
        self.update()
        self.pushButton.hide()
        self.pushButton_2.hide()
        self.pushButton_5.show()
        self.pushButton_6.show()
        self.lineEdit_3.show()
        self.lineEdit_4.show()
        self.lineEdit_5.show()
        self.lineEdit_6.show()
        self.lineEdit_7.show()

    def Login(self):
        self.showGifWindow()
        account = self.lineEdit.text()
        password = self.lineEdit_2.text()
        if account == '' or password == '':
            QMessageBox.warning(
                self, 'Warning', 'Please enter account and password.')
        else:
            try:
                self.db = mysql.connector.connect(
                    host='localhost',
                    port=3306,
                    user='root',
                    password='daidai',
                    database='123'
                )
                cursor = self.db.cursor()
                cursor.execute(
                    'SELECT * FROM member WHERE account=%s AND password=%s', (account, password))
                result = cursor.fetchone()
                if result:
                    QtWidgets.QMessageBox.information(
                        self, 'Success', 'Login successful')
                    self.pixmap = QtGui.QPixmap('pic/menu.png')
                    self.update()
                    self.pushButton_7.show()
                    self.pushButton_8.show()
                    self.pushButton_9.show()
                    self.pushButton_10.show()
                    self.pushButton_11.show()
                    self.pushButton_3.hide()
                    self.pushButton_4.hide()
                    self.lineEdit.hide()
                    self.lineEdit_2.hide()
                else:
                    QtWidgets.QMessageBox.warning(
                        self, 'Error', 'Invalid account or password')
            except mysql.connector.Error as err:
                QMessageBox.warning(self, 'Warning', str(err))
                return
            finally:
                # 確保關閉資料庫連線
                if self.db.is_connected():
                    cursor.close()
                    self.db.close()
                    self.load_grade()

    def loginback(self):
        self.pixmap = QtGui.QPixmap('pic/homepage.png')
        self.update()
        self.pushButton.show()
        self.pushButton_2.show()
        self.pushButton_3.hide()
        self.pushButton_4.hide()
        self.lineEdit.hide()
        self.lineEdit_2.hide()
        self.lineEdit.clear()
        self.lineEdit_2.clear()

    def Register(self):
        self.showGifWindow()
        account = self.lineEdit_3.text()
        password = self.lineEdit_4.text()
        name = self.lineEdit_5.text()
        gender = self.lineEdit_6.text()
        email = self.lineEdit_7.text()
        if account == '' or password == '' or name == '' or gender == '' or email == '':
            QMessageBox.warning(self, 'Warning', '請輸入完整')
        else:
            try:
                self.db = mysql.connector.connect(
                    host='localhost',
                    port=3306,
                    user='root',
                    password='daidai',
                    database='123'
                )
                cursor = self.db.cursor()
                cursor.execute(
                    'SELECT * FROM member WHERE account=%s', (account,))
                result = cursor.fetchone()
                if result:
                    QtWidgets.QMessageBox.warning(
                        self, 'Error', 'Account already exists')
                    return
                else:
                    # Insert new user into database
                    cursor.execute('INSERT INTO member (account, password,name,gender,email) VALUES (%s, %s,%s,%s,%s)', (
                        account, password, name, gender, email))
                    self.db.commit()
                    QtWidgets.QMessageBox.information(
                    self, 'Success', 'Registration successful')
                    self.pixmap = QtGui.QPixmap('pic/homepage.png')
                    self.update()
                    self.pushButton.show()
                    self.pushButton_2.show()
                    self.pushButton_5.hide()
                    self.pushButton_6.hide()
                    self.lineEdit_3.hide()
                    self.lineEdit_4.hide()
                    self.lineEdit_5.hide()
                    self.lineEdit_6.hide()
                    self.lineEdit_7.hide()
                    self.lineEdit_3.clear()
                    self.lineEdit_4.clear()
                    self.lineEdit_5.clear()
                    self.lineEdit_6.clear()
                    self.lineEdit_7.clear()
            except mysql.connector.Error as err:
                QMessageBox.warning(self, 'Warning', str(err))
                return

    def regisback(self):
        self.pixmap = QtGui.QPixmap('pic/homepage.png')
        self.update()
        self.pushButton.show()
        self.pushButton_2.show()
        self.pushButton_5.hide()
        self.pushButton_6.hide()
        self.lineEdit_3.hide()
        self.lineEdit_4.hide()
        self.lineEdit_5.hide()
        self.lineEdit_6.hide()
        self.lineEdit_7.hide()
        self.lineEdit_3.clear()
        self.lineEdit_4.clear()
        self.lineEdit_5.clear()
        self.lineEdit_6.clear()
        self.lineEdit_7.clear()
    
    def launch_apk(self, package_name):
        command = f'adb shell monkey -p {package_name} -c android.intent.category.LAUNCHER 1'
        subprocess.call(command, shell=True)

    def profile(self):
        # Load user data
        try:
            self.db = mysql.connector.connect(
                host='localhost',
                user='root',
                password='daidai',
                database='123'
            )
            cursor = self.db.cursor()
            account=self.lineEdit.text()
            cursor.execute('SELECT * FROM member WHERE account=%s', (account,))
            result = cursor.fetchone()
            if result:
                self.lineEdit_8.setText(result[0])  # account
                self.lineEdit_9.setText(result[1])  # password
                self.lineEdit_10.setText(result[2])  # name
                self.lineEdit_11.setText(result[3])  # gender
                self.lineEdit_12.setText(result[4])  # email
        except mysql.connector.Error as err:
            QMessageBox.warning(self, 'Warning', str(err))
            return
        
        self.pixmap = QtGui.QPixmap('pic/profilepage.png')
        self.update()
        self.pushButton_7.hide()
        self.pushButton_8.hide()
        self.pushButton_9.hide()
        self.pushButton_10.hide()
        self.pushButton_11.hide()
        self.pushButton_12.show()
        self.lineEdit_8.show()
        self.lineEdit_9.show()
        self.lineEdit_10.show()
        self.lineEdit_11.show()
        self.lineEdit_12.show()

    def social(self):
        self.pixmap = QtGui.QPixmap('pic/identifypage.png')
        self.update()
        self.pushButton_7.hide()
        self.pushButton_8.hide()
        self.pushButton_9.hide()
        self.pushButton_10.hide()
        self.pushButton_11.hide()
        self.tableView.show()
        self.tableView_2.show()
        self.textEdit.show()
        self.pushButton_13.show()
        self.pushButton_14.show()
        self.pushButton_15.show()
        self.pushButton_16.show()
        self.pushButton_22.show()
        self.pushButton_23.show()

    def identify(self):
        self.pixmap = QtGui.QPixmap('pic/textpage.png')
        self.update()
        self.pushButton_7.hide()
        self.pushButton_8.hide()
        self.pushButton_9.hide()
        self.pushButton_10.hide()
        self.pushButton_11.hide()
        self.pushButton_17.show()
        self.pushButton_18.show()
        self.pushButton_19.show()
        self.pushButton_20.show()
        self.pushButton_21.show()
        self.tableView_3.show()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.load_grade)
        interval = 5000
        self.timer.start(interval)           

    def menuback(self):
        self.pixmap = QtGui.QPixmap('pic/homepage.png')
        self.update()
        self.pushButton.show()
        self.pushButton_2.show()
        self.pushButton_7.hide()
        self.pushButton_8.hide()
        self.pushButton_9.hide()
        self.pushButton_10.hide()
        self.pushButton_11.hide()
        self.lineEdit.clear()
        self.lineEdit_2.clear()

    def userprofileback(self):
        self.pixmap = QtGui.QPixmap('pic/menu.png')
        self.update()
        self.pushButton_7.show()
        self.pushButton_8.show()
        self.pushButton_9.show()
        self.pushButton_10.show()
        self.pushButton_11.show()
        self.pushButton_12.hide()
        self.lineEdit_8.hide()
        self.lineEdit_9.hide()
        self.lineEdit_10.hide()
        self.lineEdit_11.hide()
        self.lineEdit_12.hide()
        self.anibutton2.hide()

    def open_add_video_dialog(self):
        account = self.lineEdit.text()
        dialog = AddVideoDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.load_videos()

    def socialback(self):
        self.pixmap = QtGui.QPixmap('pic/menu.png')
        self.update()
        self.pushButton_7.show()
        self.pushButton_8.show()
        self.pushButton_9.show()
        self.pushButton_10.show()
        self.pushButton_11.show()
        self.tableView.hide()
        self.tableView_2.hide()
        self.textEdit.hide()
        self.pushButton_13.hide()
        self.pushButton_14.hide()
        self.pushButton_15.hide()
        self.pushButton_16.hide()
        self.anibutton1.hide()
        self.pushButton_22.hide()
        self.pushButton_23.hide()

    def handle_button_click(self):
        self.selected_video_id = None

    def load_videos(self):
        self.selected_video_id=None
        try:
            self.db = mysql.connector.connect(
                host='localhost',
                user='root',
                password='daidai',
                database='123'
            )
            cursor = self.db.cursor()
            cursor.execute('SELECT video.videoid, member.name, video.title FROM video JOIN member ON video.account = member.account')
            videos = cursor.fetchall()

            model = QtGui.QStandardItemModel(len(videos), 3)
            model.setHorizontalHeaderLabels(['VideoID', 'Name', 'Title'])

            for row, video in enumerate(videos):
                for col, data in enumerate(video):
                    item = QtGui.QStandardItem(str(data))
                    item.setTextAlignment(Qt.AlignCenter)
                    model.setItem(row, col, item)

            self.tableView.setModel(model)
            self.tableView.setModel(model)
            header = self.tableView.horizontalHeader()
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
            self.tableView.setColumnWidth(1, 140)
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
            self.tableView.setColumnWidth(2, 280)
            self.tableView.hideColumn(0)  # 隱藏第一行


        except mysql.connector.Error as err:
            QtWidgets.QMessageBox.warning(self, 'Warning', str(err))
        table_view = self.tableView

        table_view.setWordWrap(True)
        header = table_view.horizontalHeader()

    def self_videos(self):
        self.selected_video_id=None
        try:
            self.db = mysql.connector.connect(
                host='localhost',
                user='root',
                password='daidai',
                database='123'
            )
            account=self.lineEdit.text()
            cursor = self.db.cursor()
            cursor.execute('SELECT video.videoid, member.name, video.title FROM video JOIN member ON video.account = member.account WHERE video.account = %s', (account,))
            videos = cursor.fetchall()

            model = QtGui.QStandardItemModel(len(videos), 3)
            model.setHorizontalHeaderLabels(['VideoID', 'Name', 'Title'])

            for row, video in enumerate(videos):
                for col, data in enumerate(video):
                    item = QtGui.QStandardItem(str(data))
                    item.setTextAlignment(Qt.AlignCenter)
                    model.setItem(row, col, item)

            self.tableView.setModel(model)
            self.tableView.setModel(model)
            header = self.tableView.horizontalHeader()
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
            self.tableView.setColumnWidth(1, 140)
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
            self.tableView.setColumnWidth(2, 280)
            self.tableView.hideColumn(0)  # 隱藏第一行


        except mysql.connector.Error as err:
            QtWidgets.QMessageBox.warning(self, 'Warning', str(err))
        table_view = self.tableView

        table_view.setWordWrap(True)
        header = table_view.horizontalHeader()

    def load_comments(self):
        try:
            self.db = mysql.connector.connect(
                host='localhost',
                user='root',
                password='daidai',
                database='123'
            )
            

            cursor = self.db.cursor()
            cursor.execute('SELECT member.name, comment.time, comment.comment FROM comment JOIN member ON comment.account = member.account WHERE comment.videoid = %s', (self.selected_video_id,))
            comments = cursor.fetchall()

            model = QtGui.QStandardItemModel(len(comments), 3)
            model.setHorizontalHeaderLabels(['Name', 'Time', 'Comment'])

            for row, comment in enumerate(comments):
                for col, data in enumerate(comment):
                    item = QtGui.QStandardItem(str(data))
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    model.setItem(row, col, item)

            self.tableView_2.setModel(model)
            header = self.tableView_2.horizontalHeader()
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
            self.tableView_2.setColumnWidth(0, 140)
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
            self.tableView_2.setColumnWidth(1, 140)
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
            self.tableView_2.setColumnWidth(2, 140)

        except mysql.connector.Error as err:
            QtWidgets.QMessageBox.warning(self, 'Warning', str(err))
        table_view = self.tableView_2

        table_view.setWordWrap(True)

    def handle_video_clicked(self, index):
        # Get selected video ID
        self.selected_video_id = int(self.tableView.model().data(index.sibling(index.row(), 0), Qt.DisplayRole))
        # Load comments for selected video
        self.load_comments()

    def submit_comment(self):
        comment_text = self.textEdit.toPlainText()

        if not comment_text:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Comment cannot be empty.')
            return

        account = self.lineEdit.text()

        try:
            cursor = self.db.cursor()
            cursor.execute('SELECT MAX(commentid) FROM comment')
            result = cursor.fetchone()
            if result[0] is None:
                commentid = 1
            else:
                commentid = result[0] + 1
            cursor = self.db.cursor()
            query = 'INSERT INTO comment (videoid, account, comment,commentid) VALUES (%s, %s, %s,%s)'
            values = (self.selected_video_id, account, comment_text,commentid)
            cursor.execute(query, values)
            self.db.commit()

            QtWidgets.QMessageBox.information(self, 'Success', 'Comment submitted successfully.')
            self.textEdit.clear()

            # Refresh comments after submitting
            self.load_comments()

        except mysql.connector.Error as err:
            QtWidgets.QMessageBox.warning(self, 'Warning', str(err))

    def handle_play(self):
        if self.selected_video_id is None:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please select a video.')
            return
        selected_row = self.tableView.selectedIndexes()[0].row()
        video_id = int(self.tableView.model().data(self.tableView.model().index(selected_row, 0), Qt.DisplayRole))
        cursor = self.db.cursor()
        cursor.execute('SELECT url FROM video WHERE videoid = %s', (video_id,))
        result = cursor.fetchone()
        if result is None:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Video not found.')
            return
        url = result[0]
        video_id = self.get_video_id_from_url(url)
        self.yt_window = YouTubeWindow(video_id)
        self.yt_window.show()

    def get_video_id_from_url(self, url):
        video_id = ''
        if 'youtube.com/watch?' in url:
            query_string = url.split('?')[1]
            parameters = query_string.split('&')
            for parameter in parameters:
                if parameter.startswith('v='):
                    video_id = parameter.split('=')[1]
                    break
        elif 'youtu.be/' in url:
            video_id = url.split('/')[-1]
        return video_id
    
    def open_upload_window(self):
        if not hasattr(self, "upload_window") or self.upload_window is None:
            self.upload_window = UploadWindow(parent=self,account=self.lineEdit.text())
        self.upload_window.exec_()

    def textback(self):
        self.pixmap = QtGui.QPixmap('pic/menu.png')
        self.update()
        self.pushButton_7.show()
        self.pushButton_8.show()
        self.pushButton_9.show()
        self.pushButton_10.show()
        self.pushButton_11.show()
        self.pushButton_17.hide()
        self.pushButton_18.hide()
        self.pushButton_19.hide()
        self.pushButton_20.hide()
        self.pushButton_21.hide()
        self.tableView_3.hide()
        self.anibutton.hide()
        self.timer.stop()  

    def load_grade(self):
        try:
            self.db = mysql.connector.connect(
                host='localhost',
                user='root',
                password='daidai',
                database='123'
            )
            account=self.lineEdit.text()
            print(account)
            cursor = self.db.cursor()
            cursor.execute('SELECT gradeid, grade, speed, time FROM grade WHERE account = %s', (account,))
            grades = cursor.fetchall()

            model = QtGui.QStandardItemModel(len(grades), 4)
            model.setHorizontalHeaderLabels(['GradeID', 'Grade', 'Speed','Time'])

            for row, video in enumerate(grades):
                for col, data in enumerate(video):
                    item = QtGui.QStandardItem(str(data))
                    item.setTextAlignment(Qt.AlignCenter)
                    model.setItem(row, col, item)

            self.tableView_3.setModel(model)
            header = self.tableView_3.horizontalHeader()
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
            self.tableView_3.setColumnWidth(1, 420)
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
            self.tableView_3.setColumnWidth(2, 280)
            header.setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)
            self.tableView_3.setColumnWidth(3, 140)
            self.tableView_3.hideColumn(0)

        except mysql.connector.Error as err:
            QtWidgets.QMessageBox.warning(self, 'Warning', str(err))

    def start_recording(self):
        self.recording = True
        self.show_waiting_window()
        self.record_audio_thread()

    def record_audio_thread(self):
        self.recording = True
        self.frames = []
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 1
        fs = 44100

        p = pyaudio.PyAudio()

        self.stream = self.p.open(format=sample_format,
                                  channels=channels,
                                  rate=fs,
                                  frames_per_buffer=chunk,
                                  input=True)

        print("錄音中...")

        def record_loop():
            while self.recording:
                data = self.stream.read(chunk)
                self.frames.append(data)

        threading.Thread(target=record_loop).start()

    def stop_recording_and_transcribe(self):
        try:
            self.recording = False

            self.waiting_window1 = WaitingWindow1(parent=self)
            
            self.waiting_window.close()
            self.waiting_window1.show()

            print("錄音完成。")
            file_name = str(random.randint(1, 1000000)) + '.wav'
            voice_folder = os.path.join(os.path.dirname(__file__), 'voice')
            if not os.path.exists(voice_folder):
                os.makedirs(voice_folder)
            file_path = os.path.join(voice_folder, file_name)
            
            if self.stream is not None:
                self.save_audio(file_path)
                text = self.transcribe(file_path)
                scores, total_score, average_score = self.evaluate_speech(text)
                speeds, total_average_speed = self.perform_speed_analysis(file_path)
                volume_data = measure_volume(file_path)
                result_window = ResultWindow(scores, total_score, average_score, speeds, total_average_speed, volume_data, self)
                suggestions = result_window.provide_suggestions(volume_data, total_average_speed)
                suggestion_text = "\n".join(suggestions)
                result_window.exec_()
                self.transcription_display.setText(text + "\n\n建議：\n" + suggestion_text)
                account = self.lineEdit.text()

                try:
                    self.db = mysql.connector.connect(
                        host='localhost',
                        user='root',
                        password='daidai',
                        database='123'
                    )
                    cursor = self.db.cursor()
                    cursor.execute('SELECT MAX(gradeid) FROM grade')
                    result = cursor.fetchone()
                    if result[0] is None:
                        gradeid = 1
                    else:
                        gradeid = result[0] + 1

                    upload_time = datetime.now()
                    cursor = self.db.cursor()
                    scores_string = ", ".join([f"{criterion}: {score}" for criterion, score in scores.items()])

                    query = 'INSERT INTO grade (gradeid, account, speech, grade, time, speed) VALUES (%s, %s, %s, %s, %s, %s)'
                    values = (gradeid, account, file_path, scores_string, upload_time, f"總平均語速為 {total_average_speed:.2f}字/秒，平均音量為 {np.mean(volume_data):.2f} dB")
                    cursor.execute(query, values)
                    self.db.commit()

                    # Refresh comments after submitting
                except mysql.connector.Error as err:
                    QtWidgets.QMessageBox.warning(self, 'Warning', str(err))
        except:
                None

        self.waiting_window1.close()

    def save_audio(self, file_path):
        channels = 1
        sample_format = pyaudio.paInt16
        fs = 44100
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        print(f"已將錄音存為 {file_path}。")

    def transcribe(self, file_path):
        try:
            openai.api_key = '0sk-tifdeUVE5oeV8RpW146NT3BlbkFJeddkwiIzdj4MprPeeQw0'
            audio_file = open(file_path, "rb")
            transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file, response_format="text")
            text = transcript.replace(" ", "")
            self.transcription_display.setText(text)

            # 讀取音頻檔案，進行降噪處理
            sound = AudioSegment.from_file(file_path, format="wav")
            # 使用低通濾波器降噪
            sound = sound.low_pass_filter(3000)
            # 使用高通濾波器降噪
            sound = sound.high_pass_filter(200)
            # 正規化音量
            sound = sound.normalize()
            # 將處理後的音頻存儲為wav格式檔案
            sound.export(file_path, format="wav")

            return text
        except RateLimitError as e:
            print(e)

    def evaluate_speech(self, speech):
        try:
            evaluation_criteria = [
                "內容完整性",
                "語言表達",
                "邏輯結構",
                "語法正確性",
                "表達力",
            ]

            scores = {
                "內容完整性": 0,
                "語言表達": 0,
                "邏輯結構": 0,
                "語法正確性":0,
                "表達力":0,
            }
            for criterion in evaluation_criteria:
                prompt = f"請您扮演一位老師，以同樣的標準幫我給以下講稿做評分：\n\n{speech}\n\n請對「{criterion}」進行評分，滿分為10分。"
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=100,
                    n=1,
                    stop=None,
                    temperature=0.8,
                )
                score_text = response.choices[0].text.strip()
                try:
                    score = int(re.search(r'\d+', score_text).group())
                    scores[criterion] = score
                    print(f"{criterion}：{score}分")
                except AttributeError:
                    print(f"未能獲取到 '{criterion}' 的分數。")

            total_score = sum(scores.values())
            if len(scores) > 0:
                average_score = total_score / len(scores)
            else:
                average_score = 0
            print(f"總分：{total_score}分")
            print(f"平均分：{average_score:.2f}分")

            return scores, total_score, average_score
        except RateLimitError as e:
            QtWidgets.QMessageBox.warning(self, "Warning", str(e))
            return

    def perform_speed_analysis(self, file_path):
        list_duration=[]
        list_duration2=[]
        Speech=[]
        Speed=[]
        if not os.path.exists('temdir2'):
            os.mkdir('temdir2')

        audiofile = AudioSegment.from_file(file_path, "wav")

        chunklist=make_chunks(audiofile, 10000)
        for i, chunk in enumerate(chunklist):
            chunk_name="temdir2/chunk{0}.wav".format(i)
            print("存檔:",chunk_name)
            chunk.export(chunk_name,format="wav")
            song = AudioSegment.from_mp3("{}".format(chunk_name)) 
            duration = song.duration_seconds
            list_duration.append(duration)
        summ=0    
        for i in range(len(list_duration)):
            summ+=list_duration[i]
            list_duration2.append(summ)
        r=sr.Recognizer()
        print("開始辨識...")
        file=open("phthon_sr.txt","w")
        for i in range(len(chunklist)):
            try:
                with sr.WavFile("temdir2/chunk{}.wav".format(i)) as source:
                    audio=r.record(source)
                result=r.recognize_google(audio,language="zh-TW")
                print("  "+str(result))
                Speech.append(result)
                file.write(result)
            except sr.UnknownValueError:
                print("Google Speech Recognition 無法辨識此語音!")
            except sr.RequestError as e:
                print("無法由 Google Speech Recognition 取得結果; {0}".format(e))
        file.close()
        print("辨識結束!")
        shutil.rmtree('temdir2')
        print(Speech)
        print(list_duration)
        print(list_duration2)  
        print(len(list_duration))

        start=0
        avgspeed=0
        totalword=0
        speeds = []
        for i in range(len(Speech)):
            avgspeed = (len(Speech[i])) / (list_duration[i])
            totalword += len(Speech[i])
            Speed.append(avgspeed)
            print(str(start) + '~' + str(start + list_duration[i]) + '秒的平均語速是' + str(avgspeed) + '字/秒')
            speeds.append((start, start + list_duration[i], avgspeed))
            start += 10
        avgspeed = (totalword) / (list_duration2[-1])
        print("0~" + str(list_duration2[-1]) + "秒的總平均語速是" + str(avgspeed) + '字/秒')
        print(Speed)
        return speeds, avgspeed

    def play_audio(self):
            if not self.tableView_3.selectedIndexes():
                QtWidgets.QMessageBox.warning(self, 'Warning', 'Please select a grade.')
                return
            selected_row = self.tableView_3.selectedIndexes()[0].row()
            grade_id = int(self.tableView_3.model().data(self.tableView_3.model().index(selected_row, 0), Qt.DisplayRole))

            try:
                self.db = mysql.connector.connect(
                    host='localhost',
                    user='root',
                    password='daidai',
                    database='123'
                )

                cursor = self.db.cursor()
                cursor.execute('SELECT speech FROM grade WHERE gradeid = %s', (grade_id,))
                result = cursor.fetchone()

                if result is None:
                    QtWidgets.QMessageBox.warning(self, 'Warning', 'Grade not found.')
                    return

                speech_path = result[0]  # 資料庫中音檔的儲存位置

                if speech_path.endswith('.wav'):
                    # 播放音檔
                    self.media_player = QMediaPlayer()
                    self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(speech_path)))
                    self.media_player.mediaStatusChanged.connect(self.handle_media_status_changed)
                    self.media_player.play()
                else:
                    QtWidgets.QMessageBox.warning(self, 'Warning', 'Invalid audio file format.')

            except mysql.connector.Error as err:
                QtWidgets.QMessageBox.warning(self, 'Warning', str(err))

    def handle_media_status_changed(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.media_player.stop()
            self.media_player = None

    def show_waiting_window(self):
        self.waiting_window = WaitingWindow(self)
        self.waiting_window.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
