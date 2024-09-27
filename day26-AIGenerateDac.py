from diagrams import Cluster, Diagram
from diagrams.aws.network import VPC, ElasticLoadBalancing, Route53
from diagrams.aws.compute import EKS
from diagrams.aws.database import Aurora, Elasticache
from diagrams.aws.storage import S3, FSx
from diagrams.aws.management import Cloudwatch
from diagrams.aws.security import IAM, KMS, SecurityHub, Shield
from diagrams.onprem.client import Users

graph_attr = {
    "fontsize": "20",
    "bgcolor": "white"
}

with Diagram("Secure Website System Architecture", show=False, graph_attr=graph_attr):
    user = Users("Users")

    with Cluster("Network", graph_attr={"bgcolor": "lightblue"}):
        vpc = VPC("VPC")
        dns = Route53("Route 53")

    with Cluster("Load Balancing", graph_attr={"bgcolor": "orange"}):
        elb = ElasticLoadBalancing("ELB")

    with Cluster("Compute", graph_attr={"bgcolor": "lightgreen"}):
        eks = EKS("EKS Cluster")

    with Cluster("Storage", graph_attr={"bgcolor": "lightyellow"}):
        s3 = S3("S3")
        fsx = FSx("FSx")

    with Cluster("Database", graph_attr={"bgcolor": "lightcoral"}):
        aurora = Aurora("Aurora")
        cache = Elasticache("Elasticache")

    with Cluster("Monitoring", graph_attr={"bgcolor": "lightgrey"}):
        cloudwatch = Cloudwatch("Cloudwatch")

    user >> dns >> elb >> eks
    eks >> s3
    eks >> fsx
    eks >> aurora
    eks >> cache
    cloudwatch >> eks

    with Cluster("Security", graph_attr={"bgcolor": "lightpink"}):
        iam = IAM("IAM")
        kms = KMS("KMS")
        securityhub = SecurityHub("SecurityHub")
        shield = Shield("Shield")

    # Security services are not connected to other services directly,
    # but they ensure the overall security of the architecture.
