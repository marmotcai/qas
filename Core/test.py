import pika

credentials = pika.PlainCredentials('guizhou_newmedia_xsyj', 'LwA5-dDrS-WKZB')
#175.6.15.87:5672, 175.6.15.136:5672, 111.8.3.77:15672, 111.8.3.77:15682
connection = pika.BlockingConnection(pika.ConnectionParameters(host = '175.6.15.87',
                                        port = 5672,virtual_host = 'cms_host',credentials = credentials))
channel = connection.channel()
# 申明消息队列，消息在这个队列传递，如果不存在，则创建队列
channel.queue_declare(queue = 'guizhou_newmedia_xsyj_assetInfo', durable = False)
# 定义一个回调函数来处理消息队列中的消息，这里是打印出来
def callback(ch, method, properties, body):
    ch.basic_ack(delivery_tag = method.delivery_tag)
    print(body.decode())

# 告诉rabbitmq，用callback来接收消息
channel.basic_consume('guizhou_newmedia_xsyj_assetInfo',callback)
# 开始接收信息，并进入阻塞状态，队列里有信息才会调用callback进行处理
channel.start_consuming()