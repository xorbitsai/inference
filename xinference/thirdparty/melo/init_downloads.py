

if __name__ == '__main__':

    from melo.api import TTS
    device = 'auto'
    models = {
        'EN': TTS(language='EN', device=device),
        'ES': TTS(language='ES', device=device),
        'FR': TTS(language='FR', device=device),
        'ZH': TTS(language='ZH', device=device),
        'JP': TTS(language='JP', device=device),
        'KR': TTS(language='KR', device=device),
    }