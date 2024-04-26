from aiortc import RTCIceServer, RTCConfiguration

stun_server = RTCIceServer(urls='WRITE HERE')
turn_server = RTCIceServer(urls='WRITE HERE', username='USER NAME', credential='CREDENTIAL')
rtc_configuration = RTCConfiguration(iceServers=[stun_server, turn_server])
