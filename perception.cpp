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


using namespace pcl;

const int nGridX = 4; // Number of grids along x (left - right)
const int nGridZ = 4; // Number of grids along z (front - back)
const float fThresh = 0.05f; // 5 cm

inline void print_point(PointXYZRGB p){
  std::cout << " X: " << p.x
            << " Y: " << p.y 
            << " Z:" << p.z << std::endl;
}

void view_point_cloud(PointCloud<PointXYZRGB>::Ptr cloud){
  visualization::CloudViewer viewer ("Cloud Viewer");
  viewer.showCloud(cloud);
  while(!viewer.wasStopped()){};
}

void downsampling(PointCloud<PointXYZRGB>::Ptr pcloud){
    // Perform downsampling with voxel
  float fleaf = 0.1f; // Determine the filter size
  VoxelGrid<PointXYZRGB> vg;
  vg.setInputCloud(pcloud);
  vg.setLeafSize(fleaf, fleaf, fleaf);
  vg.filter(*pcloud);

  std::cerr << "PointCloud after filtering: " << pcloud->points.size()
       << " data points \n";
}

void groundRemoval(PointCloud<PointXYZRGB>::Ptr pcloud, 
                   PointCloud<PointXYZRGB>::Ptr pOut){
  // Gather 3D bounds
  PointXYZRGB minPoint, maxPoint;
  getMinMax3D(*pcloud, minPoint, maxPoint);

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

  PointXYZ minBox, maxBox; // the 3D point of a box
  // Assign the values for the first grid
  minBox.x = minPoint.x; 
  minBox.y = minPoint.y; 
  minBox.z = minPoint.z;
  maxBox.y = maxPoint.y; 
  maxBox.x = minPoint.x + fGridX;
  maxBox.z = minPoint.z + fGridZ;

  PointCloud<PointXYZRGB>::Ptr gCloud (new PointCloud<PointXYZRGB>);
  PointXYZRGB localMin, localMax;

  // create filter object
  CropBox<PointXYZRGB> grid;
  grid.setInputCloud(pcloud);

  PassThrough<PointXYZRGB> pass;
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
      getMinMax3D(*gCloud, localMin, localMax);
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

int main(int argc, char** argv){
  // Start time
  std::chrono::steady_clock::time_point begin =\
  std::chrono::steady_clock::now();

  // Reading PCD file
  PointCloud<PointXYZRGB>::Ptr pcloud (new PointCloud<PointXYZRGB>);

  // read the input pcd file
  if (argc > 1){
    if (io::loadPCDFile<PointXYZRGB> (argv[1], *pcloud) == -1){ 
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
  downsampling(pcloud);

  // Ground removal
  PointCloud<PointXYZRGB>::Ptr main_cloud (new PointCloud<PointXYZRGB>);
  groundRemoval(pcloud, main_cloud);

  std::chrono::steady_clock::time_point ending =\
  std::chrono::steady_clock::now();

  std::cout << "Time difference = " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(ending - begin).count()
            << "[ms]" << std::endl;

  // View result
  view_point_cloud(pcloud);
  view_point_cloud(main_cloud);

  return (0);  
}