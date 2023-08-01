import pygame
import glob
# kor_wav = gTTS('강나훈님 안녕하세요')
# kor_wav.save('kor3.mp3')

def playTTS(names, known):

    freq = 16000    # sampling rate, 44100(CD), 16000(Naver TTS), 24000(google TTS)
    bitsize = -16   # signed 16 bit. support 8,-8,16,-16
    channels = 1    # 1 is mono, 2 is stereo
    buffer = 2048   # number of samples (experiment to get right sound)
    pygame.mixer.init(freq, bitsize, channels, buffer)

    if known:
        for i, name in enumerate(names):
            try:
                nameSound = pygame.mixer.Sound(glob.glob('./audio/name/%s*' % name)[0])
                nameSound.play()
                pygame.time.wait(int(nameSound.get_length() * 1000))
            except:
                hello = pygame.mixer.Sound('./audio/hello/안녕하세요..wav')
                hello.play()
                pygame.time.wait(int(hello.get_length() * 1000))


        welcome = pygame.mixer.Sound('./audio/hello/반갑습니다..wav')
        welcome.play()
        pygame.time.wait(int(welcome.get_length() * 2500))
    else:
        hello = pygame.mixer.Sound('./audio/hello/안녕하세요..wav')
        hello.play()
        pygame.time.wait(int(hello.get_length() * 1000))
        welcome = pygame.mixer.Sound('./audio/hello/반갑습니다..wav')
        welcome.play()
        pygame.time.wait(int(welcome.get_length() * 2500))
    # if known == False:   
    #     pygame.mixer.music.load('./audio/hello/안녕하세요..wav')
    #     pygame.mixer.music.queue('./audio/hello/반갑습니다..wav')
    #     pygame.mixer.music.play()

    # else:
    #     for i, name in enumerate(names):
    #         if i == 0:
    #             pygame.mixer.music.load('./audio/name/%s.wav' %(name + '님,'))
    #         else:
    #             pygame.mixer.music.queue('./audio/name/%s.wav' %(name + '님,'))
    # pygame.mixer.music.queue('./audio/hello/반갑습니다..wav')
    #     # pygame.mixer.music.queue('./audio/name/%s.wav' %name)
    # pygame.mixer.music.play()

    # clock = pygame.time.Clock()
    # while pygame.mixer.music.get_busy():
    #     clock.tick(3)
    # pygame.mixer.quit()    
