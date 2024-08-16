# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import inspect
import io
import os
import tempfile

import numpy as np
import pandas as pd
import torch


def test_chattts(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="ChatTTS",
        model_type="audio",
    )
    model = client.get_model(model_uid)
    input_string = (
        "chat T T S is a text to speech model designed for dialogue applications."
    )
    response = model.speech(input_string)
    assert type(response) is bytes
    assert len(response) > 0

    response = model.speech(
        input_string,
        voice="蘁淰敥欀堣盼壠垓栟爡蕬槳襞誗犆蕹伝墺悞圓諽勇詳帍嵩视啭発袂槠羢姰澯痵忚稞傣汪娏皎弔滿嫔螈访蚫牂厜槩蚻攭妈跬崳牄朷瑟亣贏棩灁绂撈紇葩懴仯脱绸爠乆丧谓柦啢梼摇傓裝挭你淼帇觸誐脳械梙燺怡摋桃瘓普憾榫趗慏或裒蕲荇撏峺膩囃臁抖忀卭荏蔚耚复腲咾攎艢祌莔熸喆刭萭惲荤卹杨枆伝哷皌往脕嫭匐懳棪螙荻材甞戅劍薧珤蓞梇穵姐焀殃或蚨蕼揱巡怵燳甠晣毂搓碩洞嫌瓴噶天畷层熧筐蕆俜栾将急葻螫偵豛范狯祕烬浉惟瑸歰薮盵把莯砳埧紞蚊壌臚苵艊勄謴秉毜垫攂櫕政亖婪冬篇儈毗暷裫丟斃粢影俲檿繳致稫琫甐嚴货諴贑柉侂猣緈娻缊毞菤唝噱垟荋忪溱袁簂讷誃畟聒乼乍圍淽堩爃诲畲硻璧怳媾褝樻袚哿汙灶簃坱樉吺濙峉昴牔皩剒瓟慭襋臀嗧築讜繥珨縔痻扥篤叫塄跹樜溫畽傑岜膡嘅唶讻罶喗津萷狭竐芬柤狦虝寎豮抿爄堭聋櫜洠爢誘襑噗賉甥显亷壖觇賈荲燒縠蛛濎亓搠薈懙墺歐曭塌牑硜摊婏岈縲罦槾樋桶裭袑惄斕廓燐恽喾礕瑑冠僺狲產縡瓦椶縄潧蜐拆榠左囲惹規现嫋揊槬桰瞿恢憲夙箝凕矀胹凹帏蛔必拹秼垑孀榊熧貂賺弣窕跨熅掽晼聬睹樓溍亀擑跅懕旆序怌譭谶拵玾蛲噪腪嚗桙崆睗臲咁炔奟糘緔叕幢檣莪覚莮搈繈袨睐绵蚪曪仓椚廠个蟛歶瘚谦茹疭喇宏痬秸捐呢戀朡歛傹眿潸揲挏吉赞課妮倾傀苢粐瘅捴姗畗窎丰恿諓嫢溧埿尌谌粰幑跲聶啋槯糈廛蓺茗楩倝腙搛烎塕繦瓩簕蔔圡偠咑樶蒓刯急犻傓姡暴玓梞婲瘽浐荒娫冰碛讀咬埡啮痩蘖哊琔篠橁虎償烬咄傏蜀媟崄毪罕棡儇抲经捺讇媤瘌炄褏伒賦臅艔袕峎氚瞿摗哀荶墉曊傚擝帣桷聈喅仭埫姷娉緑峬徰手濓斃疅猨疣厠巷枀咵劺汢寞焒劆杵礆憠薺棻奖謓掸窋犼媱纺趴氱裍取尽蟯淰湜娰栚綺瀙盛勫耓諈芍譛薄蔵事宠搩聂諘廅緿塽睡傊溏劧吕衤瀦境缾羠専忑癙箭揬儋眓摥厅漣诀粮弯昀峎翩稨祴煕砄汙伦潼蠺捱绔毦琹滘昉伏晔湭窮塃虇搏嵾胘埬寨薈藭刪澈苢宵埐撊媰溤瘱涧墵哼慰嫣趀籷薾擫朌氾塑珨綼帟侥秴共嵎暹堓蟾添弙蜁棖嵛坴殔诧祠咊穪事觙痆瞹詖臉耖秆臾彆蒺埣幘墰咽甐牤澢寱哹喈巇憀吏慬儺度籏籚捠度耫攅嚗蕬聫濣攴葭妙秓烅眘繏羺蓵犪觤狃危睑倗滩勃勓檗恉愱岔裎聜湥兡謚汞箟罴笀劅瓑諔炖冬矴菮敖痁磭乱偼戲袶粱氮侂謯傹笒跌菓焓叉竽優坄赍耆亄壦瀐柁肉撕袅嫍猺臕劁罸楈箏臰一",
    )
    assert type(response) is bytes
    assert len(response) > 0

    response = model.speech(input_string, stream=True)
    assert inspect.isgenerator(response)
    i = 0
    for chunk in response:
        i += 1
        assert type(chunk) is bytes
        assert len(chunk) > 0
    assert i > 5

    # Test openai API
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    with client.audio.speech.with_streaming_response.create(
        model=model_uid, input=input_string, voice="echo"
    ) as response:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
            response.stream_to_file(f.name)
            assert os.stat(f.name).st_size > 0


def test_gen_speaker():
    from ChatTTS.core import Chat

    df = pd.read_csv(
        "https://raw.githubusercontent.com/6drf21e/ChatTTS_Speaker/main/evaluation_results.csv"
    )

    def _f(x):
        b = base64.b64decode(x)
        bio = io.BytesIO(b)
        t = torch.load(bio, map_location="cpu")
        return t.detach().tolist()

    arr_seed_id = np.array(
        df["seed_id"].apply(lambda x: int(x.split("_")[1])), dtype=np.int16
    )
    arr_emb_data = np.array(df["emb_data"].apply(_f).tolist(), dtype=np.float16)

    assert arr_seed_id.shape == (2646,)
    assert arr_emb_data.shape == (2646, 768)

    speakers = dict(zip(arr_seed_id, arr_emb_data))
    arr = speakers[2155]
    tensor = torch.Tensor(arr)
    speaker = Chat._encode_spk_emb(tensor)
    assert len(speaker) > 400
