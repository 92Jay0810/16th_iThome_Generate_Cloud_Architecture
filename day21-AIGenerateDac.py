from diagrams import Cluster, Diagram
from diagrams.gcp.network import VPC, LoadBalancing, Armor, DNS, VPN
from diagrams.gcp.compute import KubernetesEngine
from diagrams.gcp.database import Firestore, SQL, Memorystore
from diagrams.gcp.storage import Filestore, Storage
from diagrams.gcp.operations import Monitoring
from diagrams.gcp.security import Iam, KeyManagementService, SecurityCommandCenter
from diagrams.onprem.client import Users

# 定義網站系統的安全雲端架構
with Diagram("Secure Website System Architecture", show=False):
    users = Users("Users")

    with Cluster("Network", graph_attr={"color": "lightblue"}):
        vpc = VPC("VPC")
        load_balancer = LoadBalancing("Load Balancer")
        dns = DNS("DNS")
        vpn = VPN("VPN Gateway")
        cdn = Armor("Cloud Armor")

        vpc - load_balancer
        load_balancer - cdn

    with Cluster("Compute", graph_attr={"color": "lightgreen"}):
        gke = KubernetesEngine("Kubernetes Engine")
        cdn - gke

    with Cluster("Data Storage", graph_attr={"color": "orange"}):
        storage = Storage("Storage")
        filestore = Filestore("Filestore")
        database = SQL("SQL Database")
        firestore = Firestore("Firestore")
        memorystore = Memorystore("Memorystore")

        gke - [storage, filestore, database, firestore, memorystore]

    with Cluster("Operations", graph_attr={"color": "lightyellow"}):
        monitoring = Monitoring("Monitoring")
        monitoring >> gke

    with Cluster("Security", graph_attr={"color": "red"}):
        iam = Iam("IAM")
        kms = KeyManagementService("KMS")
        security_command_center = SecurityCommandCenter(
            "Security Command Center")

        # security services are not directly linked to other services in this diagram
        # they work independently to ensure security and compliance

    users >> dns
    dns >> vpn
    vpn >> vpc
