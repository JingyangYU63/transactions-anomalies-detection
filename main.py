from pyspark.sql import SparkSession
import base64
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType, StructType, StructField, StringType 
from IPython.display import display, Image

# define a UDF to decode the Base64 image data
def decode_image(base64_data):
    binary_data = base64.b64decode(base64_data)
    return binary_data

# register the UDF
udf_decode_image = udf(decode_image, BinaryType())

# define the schema of the PySpark DataFrame
schema = StructType([
    StructField("image", StructType([
        StructField("data", BinaryType(), True), \
        StructField("mime-type", StringType(), True)
    ]), True)
])

sparks = SparkSession \
    .builder \
    .appName("Test") \
    .config("spark.jars.packages", "com.databricks:spark-xml_2.12:0.12.0") \
    .getOrCreate()

# read the XML file into a PySpark DataFrame
df = sparks.read.format('xml') \
    .option("rootTag", "document") \
    .option("rowTag", "document") \
    .schema(schema) \
    .load("/Users/yujingyang/Desktop/example_pic.xml")

df.printSchema()
df.show()
# extract the image data and decode it
print(df.select("image.data").collect())
image_data = df.select("image.data").collect()[0][0]
binary_data = decode_image(image_data)

# create a PySpark DataFrame that contains the image data
df_image = sparks.createDataFrame([(binary_data,)], ["image"])
    
# display the image using the display() method
display(Image(data=bytearray(binary_data)))