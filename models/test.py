import paddle
import paddle.fluid as fluid  

def reader():  
    for i in range(10):  
        yield i  

print(len( list(reader()) ))
# batch_reader = paddle.batch(reader, batch_size=2)  
# batch_reader = paddle.batch(
#     paddle.reader.shuffle(reader, 10), 2
# )

# for data in batch_reader():  
#     print(data)

# for data in batch_reader():  
#     print(data)