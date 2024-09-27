from diagrams import Cluster, Diagram, Edge
from diagrams.aws.network import ELB
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDSPostgresqlInstance
from diagrams.aws.integration import SQS
from diagrams.aws.analytics import ManagedStreamingForKafka as MSK
from diagrams.aws.analytics import Glue
from diagrams.aws.storage import S3
from diagrams.aws.security import Cognito
from diagrams.aws.management import Cloudwatch

with Diagram(name="Advanced Web Service with AWS (colored)", show=False):
    ingress = ELB("Load Balancer")

    metrics = Cloudwatch("CloudWatch")
    metrics << Edge(color="firebrick",
                    style="dashed") << Cloudwatch("Monitoring")

    with Cluster("Service Cluster"):
        grpcsvc = [
            EC2("grpc1"),
            EC2("grpc2"),
            EC2("grpc3")]

    with Cluster("Sessions HA"):
        primary = SQS("SQS Queue")
        primary \
            - Edge(color="brown", style="dashed") \
            - SQS("SQS Queue Replica") \
            << Edge(label="collect") \
            << metrics
        grpcsvc >> Edge(color="brown") >> primary

    with Cluster("Database HA"):
        primary = RDSPostgresqlInstance("RDS Postgres")
        primary \
            - Edge(color="brown", style="dotted") \
            - RDSPostgresqlInstance("RDS Replica") \
            << Edge(label="collect") \
            << metrics
        grpcsvc >> Edge(color="black") >> primary

    aggregator = Glue("Glue Crawler")
    aggregator \
        >> Edge(label="parse") \
        >> MSK("Kafka (MSK)") \
        >> Edge(color="black", style="bold") \
        >> S3("S3 Analytics Storage")

    ingress \
        >> Edge(color="darkgreen") \
        << grpcsvc \
        >> Edge(color="darkorange") \
        >> aggregator
