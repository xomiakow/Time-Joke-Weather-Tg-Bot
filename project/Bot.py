import argparse
import asyncio
import logging
import os
import random
import colorlog
import flag as flg
import pymorphy2
import requests
import uvicorn

from ConstantData import ConstDat as cD
from datetime import datetime
from aiogram import Bot, Dispatcher, types, F
from aiogram.client.session import aiohttp
from aiogram.types import Message
from fastapi import FastAPI, Response
from fuzzywuzzy import process
from joblib import load
from natasha import Doc, Segmenter, NewsEmbedding, NewsNERTagger
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from Classification import prepare


# Loading MODEL
cur_dir = os.getcwd()
model = os.path.join(cur_dir, 'trained_model.joblib')
pred_mod = load(model)


# Setting LOGGER
logging.basicConfig(level=logging.DEBUG)
logger = colorlog.getLogger('base')
parser = argparse.ArgumentParser(description='creating parser')
parser.add_argument("--loglvl", help="logging levels")
args = parser.parse_args()


# Setting LOGGING LEVELS
if args.loglvl:
    loglevel = logging.getLogger("base")
    log = args.logl
    if log.lower() == "debug":
        loglevel.setLevel(logging.DEBUG)
    elif log.lower() == "info":
        loglevel.setLevel(logging.INFO)
    elif log.lower() == "warning":
        loglevel.setLevel(logging.WARNING)

# KEYS & TOKENS

TOKEN = os.getenv('TOKEN')
WAKEY = os.getenv('WAKEY')
JOKETOKEN = os.getenv('JOKETOKEN')
JOKEPID = os.getenv('JOKEPID')


# Creating ESSENTIAL OBJECTS
sgmnt = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)
morph = pymorphy2.MorphAnalyzer()
bot = Bot(token=TOKEN)
fp = FastAPI()
dp = Dispatcher()


# Class for VALIDATING currency data
class Currency(BaseModel):
    code: str
    name: str
    value: float


# Class for getting, updating and sending to API current CURRENCY LIST
class CurrencyData:
    async def update_currencies(self):
        while True:
            for cur in cD.cur_list:
                chkcur = await self.get_cur_value(cur)
                cD.cur_data_list[cur] = {'name': chkcur.name, 'value': chkcur.value}
                logging.debug('Произошел запрос данных по валюте ' + cur)
            await asyncio.sleep(3000)

    @staticmethod
    async def get_cur_value(code):
        data = requests.get('https://www.cbr-xml-daily.ru/daily_json.js').json()
        logging.info('Подгруженны данные API')
        value = float(data['Valute'][code]['Value'] / data['Valute'][code]['Nominal'])
        return Currency(
            code=code,
            name=data['Valute'][code]['Name'],
            value=f'{value:.3f}'
        )


# Class for getting LOCATION and WEATHER from user request
class WeatherData:
    @staticmethod
    async def get_location(msg):
        msg = msg.title()
        doc = Doc(msg)
        doc.segment(sgmnt)
        doc.tag_ner(ner_tagger)
        logger.debug('Итог таггера запроса: ' + str(doc.ner))
        for span in doc.ner.spans:
            if span.type == 'LOC':
                loc = morph.parse(msg[span.start:span.stop])[0].normal_form
                logger.debug('Погода будет предоставлена по локации ' + loc)
                return loc.title()
            else:
                logger.debug('В запросе пользователя не обнаружено названия локации')
                return None

    @staticmethod
    async def get_weather(msg):
        loc = await WeatherData.get_location(msg)
        if loc is not None:
            data = requests.get('http://api.openweathermap.org/data/2.5/weather?q=' + loc +
                                '&lang=ru&units=metric&appid=' + WAKEY).json()
            if data['cod'] != 200:
                logger.debug('Локация "'+loc+'" не найдена в API')
                return 'Простите, такого места я не знаю 😳'

            city = str(morph.parse(loc)[0].inflect({'loct'}).word).capitalize()
            clouds = str(data['weather'][0]['description']).capitalize()
            temperature = float(data['main']['temp'])
            wind = str(data['wind']['speed']) + 'м/с'
            weather_state = data['weather'][0]['main']
            sunset = int(data['sys']['sunset'])
            sunrise = int(data['sys']['sunrise'])
            now = int(data['dt'])
            try:
                flag = flg.flag(data['sys']['country'])
                logger.debug('Флаг локации обнаружен')
            except KeyError:
                flag = ''
                logger.debug('Флаг локации НЕ обнаружен')
            if temperature > 25:
                temperature = str(temperature) + '°C 🥵'
            elif temperature < 15:
                temperature = str(temperature) + '°C 🥶'
            else:
                temperature = str(temperature) + '°C 😁'
            match weather_state:
                case 'Thunderstorm':
                    weather_state = '⚡️'
                case 'Rain':
                    weather_state = '💧'
                case 'Drizzle':
                    weather_state = '💦'
                case 'Snow':
                    weather_state = '❄️'
                case 'Mist':
                    weather_state = '🌁'
                case 'Smoke':
                    weather_state = '🌫'
                case 'Haze':
                    weather_state = '🌫'
                case 'Fog':
                    weather_state = '🌁'
                case 'Squall':
                    weather_state = '🌪️'
                case 'Tornado':
                    weather_state = '🌪️'
                case 'Clear':
                    if sunrise < now < sunset:
                        weather_state = '🌞'
                    else:
                        weather_state = '🌛'
                case 'Clouds':
                    weather_state = '☁️'
            logger.info('Бот сообщил погоду')
            return ('Синоптики сообщают что на данный момент в ' + flag + city + '\n\nПогода: ' + clouds + ' ' +
                    weather_state + '\nТемпература: ' + temperature + '\nСкорость ветра: ' + wind)
        else:
            return 'Простите, не понял Ваш вопрос😓'


# Class for giving user random east-european JOKES
class Joke:
    @staticmethod
    async def get_joke():
        rand = random.randint(0, len(cD.country_codes) - 1)
        joke_cntr = 'country=' + str(cD.country_codes[rand])
        data = requests.get('http://anecdotica.ru/api?pid=' + JOKEPID +
                            '&method=getRandItem&' + joke_cntr +'&token=' + JOKETOKEN).json()
        if data['result']['error'] != 0:
            logger.debug('НЕ удалось получить анекдот из API')
            return 'Анектода сегодня не будет, простите (вероятно подписка на API  санекдотами истекла)'
        logger.debug('Анекдот из API получен')
        return 'Для Вас анекдот из великой ' + cD.joke_flag[rand] + '\n' + data['item']['text']


# API for sending currencies to bot
class MyCurAPI:
    @staticmethod
    async def send_request(param):
        while True:
            fpurl = f'http://localhost:8008/' + param
            async with aiohttp.ClientSession() as session:
                async with session.get(fpurl) as resp:
                    if resp.status == 200:
                        logger.debug('Бот подключен')
                        actResp = await resp.text()
                        return actResp
                    else:
                        logger.warning('Не удается установить соединение с сервером')


#GETTERS FROM API functions start from here-----------------------------------------------------------------------------
@fp.get('/')
async def root():
    logger.info('Переход в рут')
    return cD.cur_list


@fp.get('/EUR')
async def root():
    logger.info('Переход на /EUR')
    asccur = cD.cur_data_list['EUR']
    resp = 'На текущий момент\n1 🇪🇺' + asccur['name'] + ' = ' + str(asccur['value']) + ' 🇷🇺 Российских рублей'
    return Response(content=resp, media_type="text/plain")


@fp.get('/BYN')
async def root():
    logger.info('Переход на /BYN')
    asccur = cD.cur_data_list['BYN']
    resp = 'На текущий момент\n1 🇧🇾' + asccur['name'] + ' = ' + str(asccur['value']) + ' 🇷🇺 Российских рублей'
    return Response(content=resp, media_type="text/plain")


@fp.get('/UAH')
async def root():
    logger.info('Переход на /UAH')
    asccur = cD.cur_data_list['UAH']
    resp = 'На текущий момент\n1 🇺🇦' + asccur['name'] + ' = ' + str(asccur['value']) + ' 🇷🇺 Российских рублей'
    return Response(content=resp, media_type="text/plain")


@fp.get('/MDL')
async def root():
    logger.info('Переход на /MDL')
    asccur = cD.cur_data_list['MDL']
    resp = 'На текущий момент\n1 🇲🇩' + asccur['name'] + ' = ' + str(asccur['value']) + ' 🇷🇺 Российских рублей'
    return Response(content=resp, media_type="text/plain")


@fp.get('/RON')
async def root():
    logger.info('Переход на /RON')
    asccur = cD.cur_data_list['RON']
    resp = 'На текущий момент\n1 🇷🇴' + asccur['name'] + ' = ' + str(asccur['value']) + ' 🇷🇺 Российских рублей'
    return Response(content=resp, media_type="text/plain")


@fp.get('/BGN')
async def root():
    logger.info('Переход на /BGN')
    asccur = cD.cur_data_list['BGN']
    resp = 'На текущий момент\n1 🇧🇬' + asccur['name'] + ' = ' + str(asccur['value']) + ' 🇷🇺 Российских рублей'
    return Response(content=resp, media_type="text/plain")


@fp.get('/HUF')
async def root():
    logger.info('Переход на /HUF')
    asccur = cD.cur_data_list['HUF']
    resp = 'На текущий момент\n1 🇭🇺' + asccur['name'] + ' = ' + str(asccur['value']) + ' 🇷🇺 Российских рублей'
    return Response(content=resp, media_type="text/plain")


@fp.get('/CZK')
async def root():
    logger.info('Переход на /CZK')
    asccur = cD.cur_data_list['CZK']
    resp = 'На текущий момент\n1 🇨🇿' + asccur['name'] + ' = ' + str(asccur['value']) + ' 🇷🇺 Российских рублей'
    return Response(content=resp, media_type="text/plain")


@fp.get('/PLN')
async def root():
    logger.info('Переход на /PLN')
    asccur = cD.cur_data_list['PLN']
    resp = 'На текущий момент\n1 🇵🇱' + asccur['name'] + ' = ' + str(asccur['value']) + ' 🇷🇺 Российских рублей'
    return Response(content=resp, media_type="text/plain")
#GETTERS FROM API functions end from here-------------------------------------------------------------------------------


#EXACT CURRENCY BUTTONS functions start here----------------------------------------------------------------------------
@dp.message(F.text.in_(['🇪🇺 Евро']))
async def req_eur(message: Message):
    if cD.user_choose["choose_act"] == 1:
        logger.info('Пользователь нажал кнопку "Евро"')
        sender = MyCurAPI()
        resp = await sender.send_request('EUR')
        await message.answer(resp)


@dp.message(F.text.in_(['🇧🇾 Белорусский рубль']))
async def req_byn(message: Message):
    if cD.user_choose["choose_act"] == 1:
        logger.info('Пользователь нажал кнопку "Белорусский рубль"')
        sender = MyCurAPI()
        resp = await sender.send_request('BYN')
        await message.answer(resp)


@dp.message(F.text.in_(['🇺🇦 Украинская гривна']))
async def req_uah(message: Message):
    if cD.user_choose["choose_act"] == 1:
        logger.info('Пользователь нажал кнопку "Украинская гривна"')
        sender = MyCurAPI()
        resp = await sender.send_request('UAH')
        await message.answer(resp)


@dp.message(F.text.in_(['🇲🇩 Молдавский лей']))
async def req_mdl(message: Message):
    if cD.user_choose["choose_act"] == 1:
        logger.info('Пользователь нажал кнопку "Молдавский лей"')
        sender = MyCurAPI()
        resp = await sender.send_request('MDL')
        await message.answer(resp)


@dp.message(F.text.in_(['🇷🇴 Румынский лей']))
async def req_ron(message: Message):
    if cD.user_choose["choose_act"] == 1:
        logger.info('Пользователь нажал кнопку "Румынский лей"')
        sender = MyCurAPI()
        resp = await sender.send_request('RON')
        await message.answer(resp)


@dp.message(F.text.in_(['🇧🇬 Болгарский лев']))
async def req_bgn(message: Message):
    if cD.user_choose["choose_act"] == 1:
        logger.info('Пользователь нажал кнопку "Болгарский лев"')
        sender = MyCurAPI()
        resp = await sender.send_request('BGN')
        await message.answer(resp)


@dp.message(F.text.in_(['🇭🇺 Венгерский форинт']))
async def req_huf(message: Message):
    if cD.user_choose["choose_act"] == 1:
        logger.info('Пользователь нажал кнопку "Венгерский форинт"')
        sender = MyCurAPI()
        resp = await sender.send_request('HUF')
        await message.answer(resp)


@dp.message(F.text.in_(['🇨🇿 Чешская крона']))
async def req_czk(message: Message):
    if cD.user_choose["choose_act"] == 1:
        logger.info('Пользователь нажал кнопку "Чешская крона"')
        sender = MyCurAPI()
        resp = await sender.send_request('CZK')
        await message.answer(resp)


@dp.message(F.text.in_(['🇵🇱 Польский злотый']))
async def req_pln(message: Message):
    if cD.user_choose["choose_act"] == 1:
        logger.info('Пользователь нажал кнопку "Польский злотый"')
        sender = MyCurAPI()
        resp = await sender.send_request('PLN')
        await message.answer(resp)
#EXACT CURRENCY BUTTONS functions end here------------------------------------------------------------------------------


# START CHAT function
@dp.message(F.text.in_(['Привет', 'привет', 'старт', '/start']))
async def hello(message: types.Message):
    logger.debug('Начало чата с пользоателем')
    bttns = [
        [types.KeyboardButton(text='Уровень валют восточной Европы')],
        [types.KeyboardButton(text='Заявка в УК')],
        [types.KeyboardButton(text='Другой вопрос')]
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=bttns, resize_keyboard=True)
    await message.answer('Привет!😄\nВыбери что тебя интересует👇', reply_markup=keyboard)


# CURRENCIES LIST button function
@dp.message(F.text.in_(['Уровень валют восточной Европы', 'уровень валют восточной европы', 'валюты', 'уровень валют']))
async def currencies(message: types.Message):
    if cD.user_choose["choose_act"] == 0 or 2 or 3:
        cD.user_choose["choose_act"] = 1
        logger.debug('переход в меню 1 (валюты)')
        bttns = [
            [types.KeyboardButton(text='🇪🇺 Евро')],
            [types.KeyboardButton(text='🇧🇾 Белорусский рубль')],
            [types.KeyboardButton(text='🇺🇦 Украинская гривна')],
            [types.KeyboardButton(text='🇲🇩 Молдавский лей')],
            [types.KeyboardButton(text='🇷🇴 Румынский лей')],
            [types.KeyboardButton(text='🇧🇬 Болгарский лев')],
            [types.KeyboardButton(text='🇭🇺 Венгерский форинт')],
            [types.KeyboardButton(text='🇨🇿 Чешская крона')],
            [types.KeyboardButton(text='🇵🇱 Польский злотый')],
            [types.KeyboardButton(text='↩️ Назад')]
        ]
        keyboard = types.ReplyKeyboardMarkup(keyboard=bttns, resize_keyboard=True)
        await message.answer('Выберите интересующую валюту', reply_markup=keyboard)


#BACK button function
@dp.message(F.text.in_(['↩️ Назад', 'Назад', 'назад']))
async def back(message: types.Message):
    if cD.user_choose["choose_act"] == 1 or 2 or 3:
        cD.user_choose["choose_act"] = 0
        logger.debug('Возврат к меню 0 (главное меню)')
        bttns = [
            [types.KeyboardButton(text='Уровень валют восточной Европы')],
            [types.KeyboardButton(text='Заявка в УК')],
            [types.KeyboardButton(text='Другой вопрос')]
        ]
        keyboard = types.ReplyKeyboardMarkup(keyboard=bttns, resize_keyboard=True)
        await message.answer('Доступные действия:', reply_markup=keyboard)


#OTHER QUESTIONS button function
@dp.message(F.text.in_(['Другой вопрос', 'другой вопрос', 'другое']))
async def ask_anything(message: types.Message):
    if cD.user_choose["choose_act"] == 0 or 1 or 2:
        cD.user_choose["choose_act"] = 3
        logger.debug('переход в меню 3 (другой вопрос)')
        bttns = [[types.KeyboardButton(text='↩️ Назад')]]
        keyboard = types.ReplyKeyboardMarkup(keyboard=bttns, resize_keyboard=True)
        await message.answer('Могу подсказать:\nСколько время🕰\nКакая погода в Перми и других городах🌤\nА так же '
                             'рассказать восточноевропейский анекдот🤡🇷🇸', reply_markup=keyboard)


#QUERRY button function
@dp.message(F.text.in_(['Заявка в УК']))
async def ask_for_query(message: Message):
    if cD.user_choose['choose_act'] == 0 or 1 or 3:
        cD.user_choose['choose_act'] = 2
        logger.debug('переход в меню 2 (заявка в ук)')
        bttns = [[types.KeyboardButton(text='↩️ Назад')]]
        keyboard = types.ReplyKeyboardMarkup(keyboard=bttns, resize_keyboard=True)
        await message.answer('Пожалуйста, напишите ваше обращение для управляющей компании в чат',
                             reply_markup=keyboard)


#FREE-FORM & GARBAGE QUESTIONS button function
@dp.message()
async def other_questions(message: types.Message):
    if cD.user_choose["choose_act"] == 3:
        logger.info('Обработка пользовательского вопроса в свободной форме')
        t = process.extractOne(message.text, cD.time_question)
        w = process.extractOne(message.text, cD.weather_question)
        wp = process.extractOne(message.text, cD.weather_locale_question)
        j = process.extractOne(message.text, cD.joke_question)
        logger.debug('Схожесть вопроса по Левенштейну:\nВремя - ' + str(t[1]) + '\nПогода локально - ' + str(wp[1]) +
                     '\nПогода в другом месте - ' + str(w[1]) + '\nАнекдот - ' + str(j[1]))
        if t[1] >= 80:
            await message.answer(str(datetime.now().strftime('Текущее время по Перми - %H ч. %M мин.')))
            logger.info('Бот сообщил время')
        elif wp[1] > 95:
            wd = await WeatherData.get_weather('Какая погода в Перми?')
            await message.answer(wd)
        elif w[1] >= 80:
            wd = await WeatherData.get_weather(message.text)
            await message.answer(wd)
        elif j[1] >= 80:
            jk = await Joke.get_joke()
            await message.answer(jk)
            if jk is not 'Анектода сегодня не будет, простите (вероятно подписка на API  санекдотами истекла)':
                logger.info('Бот рассказал анекдот')
            else:
                logger.info('Бот не смог выдать анекдот')
        else:
            await message.answer('Простите, на данный вопрос ответить я не могу 😓')
            logger.debug('Бот не уловил суть вопроса')
    elif cD.user_choose['choose_act'] == 2:
        query = message.text
        bttns = [[types.KeyboardButton(text='↩️ Назад')]]
        logger.info('Пользователь отправил заявку о проблемах в управляющую компанию')
        keyboard = types.ReplyKeyboardMarkup(keyboard=bttns, resize_keyboard=True)
        query = prepare(query)
        query_stat = pred_mod.predict([query])
        logger.debug('Заявке присвоен тип')
        await message.answer(f'Заявке присвоен тип: {query_stat[0]}.\nСпасибо что обратились, уже работаем над'
                             f' вашей проблемой!', reply_markup=keyboard)
    else:
        await message.answer('Простите, на данный вопрос ответить я не могу 😓')


#Building ORDER OF EXECUTION functions
async def main():
    check = CurrencyData()
    asyncio.create_task(check.update_currencies())
    asyncio.create_task(dp.start_polling(bot))
    config = uvicorn.Config(fp, host='localhost', port=8008)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == '__main__':
    asyncio.run(main())
