# -*- coding: utf-8 -*-

# %reset

"""
Created on Mon Sep  2 19:22:14 2019

@author: saihei
"""
# https://tiedostopalvelu.maanmittauslaitos.fi/tp/kartta/

import numpy as np
from xml.dom import minidom


#%%

my_file = './data/paikannimet_2019_09/paikannimi.xml'

mydoc = minidom.parse(my_file)

mynodes = mydoc.nodeName

mydoc.firstChild.tagName

myfeatures = mydoc.getElementsByTagName('gml:FeatureCollection')

mykirj = mydoc.getElementsByTagName('pnr:kirjoitusasu')

paikannimet = []
for i in range(len(mykirj)):
    nimi_tmp = mykirj[i].toxml()
    paikannimet.append(nimi_tmp)

np.save('paikannimet_xml.npy', paikannimet)

#%%

#for i in range(20):
#    print(mydoc.documentElement.childNodes.item(i).toxml())


"""<gml:featureMember>
<pnr:Paikannimi gml:id="PN_40000006">
<gml:boundedBy>
<gml:Envelope srsName="EPSG:3067">
<gml:lowerCorner>198933.728 6818193.395</gml:lowerCorner>
<gml:upperCorner>198933.728 6818193.395</gml:upperCorner>
</gml:Envelope>
</gml:boundedBy>
<pnr:paikkaID>10000006</pnr:paikkaID>
<pnr:paikkatyyppiKoodi>350</pnr:paikkatyyppiKoodi>
<pnr:paikkatyyppiryhmaKoodi>1</pnr:paikkatyyppiryhmaKoodi>
<pnr:paikkatyyppialaryhmaKoodi>11</pnr:paikkatyyppialaryhmaKoodi>
<pnr:paikkaSijainti>
<gml:Point srsName="EPSG:3067">
<gml:pos>198933.728 6818193.395</gml:pos>
</gml:Point>
</pnr:paikkaSijainti>
<pnr:paikkaKorkeus>0</pnr:paikkaKorkeus>
<pnr:tm35Fin7Koodi>M3233D3</pnr:tm35Fin7Koodi>
<pnr:ylj7Koodi>114104D</pnr:ylj7Koodi>
<pnr:pp6Koodi>21M3A4</pnr:pp6Koodi>
<pnr:kuntaKoodi>051</pnr:kuntaKoodi>
<pnr:seutukuntaKoodi>041</pnr:seutukuntaKoodi>
<pnr:maakuntaKoodi>04</pnr:maakuntaKoodi>
<pnr:laaniKoodi>2</pnr:laaniKoodi>
<pnr:suuralueKoodi>3</pnr:suuralueKoodi>
<pnr:mittakaavarelevanssiKoodi>50000</pnr:mittakaavarelevanssiKoodi>
<pnr:paikkaLuontiAika>2008-12-06T00:00:00.000</pnr:paikkaLuontiAika>
<pnr:paikkaMuutosAika>2016-12-30T11:31:22.000</pnr:paikkaMuutosAika>
<pnr:paikannimiID>40000006</pnr:paikannimiID>
<pnr:kirjoitusasu>Rahakari</pnr:kirjoitusasu>
<pnr:kieliKoodi>fin</pnr:kieliKoodi>
<pnr:kieliVirallisuusKoodi>1</pnr:kieliVirallisuusKoodi>
<pnr:kieliEnemmistoKoodi>1</pnr:kieliEnemmistoKoodi>
<pnr:paikannimiLahdeKoodi>1</pnr:paikannimiLahdeKoodi>
<pnr:paikannimiStatusKoodi>5</pnr:paikannimiStatusKoodi>
<pnr:paikannimiLuontiAika>2008-12-06T00:00:00.000</pnr:paikannimiLuontiAika>
<pnr:paikannimiMuutosAika>2008-12-06T00:00:00.000</pnr:paikannimiMuutosAika>
</pnr:Paikannimi>
</gml:featureMember>"""

