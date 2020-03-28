#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>

// For downsampling
#include <pcl/filters/voxel_grid.h>

// For elapsed time
#include <chrono>

// For ground removal
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>

// For Euclidean Clustering
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

// For image extraction
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>

const int nGridX = 6; // Number of grids along x (left - right)
const int nGridZ = 6; // Number of grids along z (front - back)
const float fThresh = 0.08f; // in meter

inline void print_point(pcl::PointXYZRGB p){
  std::cout << " X: " << p.x
            << " Y: " << p.y 
            << " Z:" << p.z << std::endl;
}

void view_point_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
  pcl::visualization::CloudViewer viewer ("Cloud Viewer");
  viewer.showCloud(cloud);
  while(!viewer.wasStopped()){};
}

void downsampling(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud,
                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sample){
    // Perform downsampling with voxel
  float fleaf = 0.02f; // Determine the filter size
  pcl::VoxelGrid<pcl::PointXYZRGB> vg;
  vg.setInputCloud(pcloud);
  vg.setLeafSize(fleaf, fleaf, fleaf);
  vg.filter(*sample);

  std::cerr << "PointCloud after filtering: " << pcloud->points.size()
       << " data points \n";
}

void ground_removal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud, 
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pOut){
  // Gather 3D bounds
  pcl::PointXYZRGB minPoint, maxPoint;
  pcl::getMinMax3D(*pcloud, minPoint, maxPoint);

  std::cout << "min point: ";
  print_point(minPoint);
  std::cout << "max point: ";
  print_point(maxPoint);

  // Calc the size of a grid
  float fGridX = (maxPoint.x - minPoint.x) / nGridX;
  float fGridZ = (maxPoint.z - minPoint.z) / nGridZ;

  // Get absolute values
  fGridX = (fGridX > 0)? fGridX : (-1 * fGridX);
  fGridZ = (fGridZ > 0)? fGridZ : (-1 * fGridZ);

  std::cout << "GridWidth: " << fGridX 
            << " GridLength: " << fGridZ << std::endl;

  pcl::PointXYZ minBox, maxBox; // the 3D point of a box
  // Assign the values for the first grid
  minBox.x = minPoint.x; 
  minBox.y = minPoint.y; 
  minBox.z = minPoint.z;
  maxBox.y = maxPoint.y; 
  maxBox.x = minPoint.x + fGridX;
  maxBox.z = minPoint.z + fGridZ;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr gCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointXYZRGB localMin, localMax;

  // create filter object
  pcl::CropBox<pcl::PointXYZRGB> grid;
  grid.setInputCloud(pcloud);

  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud(gCloud);
  pass.setFilterFieldName("y");

  // Start from min point construct each grid
  for (int xc = 0; xc < nGridX; ++xc){
    for (int zc = 0; zc < nGridZ; ++zc){  
      // For each sector, filter the lower layer
      
      grid.setMin(minBox.getVector4fMap());
      grid.setMax(maxBox.getVector4fMap());
      grid.filter(*gCloud);

      // Skip if no points
      //if (gCloud->points.size() == 0){ continue; }

      // Find minimum point
      pcl::getMinMax3D(*gCloud, localMin, localMax);
      // Keep everything between lower thresh hold and 5m from zero plane
      pass.setFilterLimits(localMin.y + fThresh, 1); 
      pass.filter(*gCloud);

      //view_point_cloud(gCloud);
      *pOut += *gCloud;

      // move the grid point along z
      minBox.z += fGridZ; 
      maxBox.z += fGridZ;
    }
    // move the grid point along x
    minBox.x += fGridX;
    maxBox.x += fGridX;

    // rest z coord
    minBox.z = minPoint.z;
    maxBox.z = minPoint.z + fGridZ;
  }
}

void euclidean_cluster(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                      std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& collection){
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud(cloud);

  // All the things GOD only knows
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance(0.1); // with in 10 cm?
  ec.setMinClusterSize(5);
  ec.setMaxClusterSize(100);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  // Make point cloud for each cluster
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
      it != cluster_indices.end(); ++it){
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    for (std::vector<int>::const_iterator pit = it->indices.begin(); 
        pit != it->indices.end(); ++pit){
      cloud_cluster->points.push_back (cloud->points[*pit]);
    }
    cloud_cluster->width = cloud_cluster->points.size();
    cloud_cluster->height= 1;
    cloud_cluster->is_dense = true;

    collection.push_back(cloud_cluster);
  }
}

void project_to_image(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr projection (new pcl::PointCloud<pcl::PointXYZRGB>);

  // Create a set of planar coeff with x=y=0 z = 1
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  coefficients->values.resize (4);
  coefficients->values[0] = coefficients->values[1] = 0;
  coefficients->values[2] = 1.0;
  coefficients->values[3] = 0;

  // Create the filtering object
  pcl::ProjectInliers<pcl::PointXYZ> proj;
  proj.setModelType (pcl::SACMODEL_PLANE);
  proj.setInputCloud (cloud);
  proj.setModelCoefficients (coefficients);
  proj.filter (*projection);

  view_point_cloud(projection);
}

int main(int argc, char** argv){
  // Start time
  std::chrono::steady_clock::time_point begin =\
  std::chrono::steady_clock::now();

  // The original input/file
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  // Downsampled point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sample (new pcl::PointCloud<pcl::PointXYZRGB>);
  // Ground removed & clustering
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr kcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  // Cluster List
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clust_collection;

  // Read the input pcd file
  if (argc > 1){
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (argv[1], *pcloud) == -1){ 
      // load the file
      PCL_ERROR( "Couldn't read file. \n");
      return (-1);
    }
  }else{
    std::cout << "No input .pcd file!" << std::endl;
    return (0);
  }

  std::cout << "Loaded "
            << pcloud->width * pcloud ->height
            << " data points from the test.pcd with the following fields: "
            << std::endl;

  // Downsampling
  downsampling(pcloud, sample);

  // Ground removal
  ground_removal(sample, kcloud);

  // Euclidean Clustering
  // Clusters will be append into cluster_colletion
  euclidean_cluster(kcloud, clust_collection);

  // Project each cluster to an image
  for (size_t i = 0; i < clust_collection.size(); ++i)
    project_to_image(clust_collection[i]);


  std::chrono::steady_clock::time_point ending =\
  std::chrono::steady_clock::now();

  std::cout << "Time difference = " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(ending - begin).count()
            << "[ms]" << std::endl;

  std::cout << "Number of clusters:: " << clust_collection.size() << std::endl;

  // View result
  view_point_cloud(pcloud);
  view_point_cloud(kcloud);

  return (0);  
}