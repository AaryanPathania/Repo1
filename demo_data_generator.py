import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random

class SyntheticBrainGenerator:
    """
    Generates synthetic brain MRI data for demonstration purposes.
    Creates 3D volumes that simulate brain structures with different patterns for AD vs Normal cases.
    """
    
    def __init__(self, output_dir="./demo_data"):
        self.output_dir = output_dir
        self.image_shape = (121, 145, 121)  # Standard brain MRI dimensions from the original project
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def create_brain_base(self):
        """Create a basic brain-like structure"""
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, self.image_shape[0]),
            np.linspace(-1, 1, self.image_shape[1]),
            np.linspace(-1, 1, self.image_shape[2]),
            indexing='ij'
        )
        
        # Create ellipsoid brain shape
        brain_mask = (x**2/0.8**2 + y**2/1.0**2 + z**2/0.7**2) < 1
        
        # Add brain tissue intensity
        brain = np.zeros(self.image_shape)
        brain[brain_mask] = 0.7 + 0.3 * np.random.random(np.sum(brain_mask))
        
        # Add ventricles (darker regions)
        ventricle_mask = (x**2/0.2**2 + y**2/0.3**2 + z**2/0.2**2) < 1
        brain[ventricle_mask] = 0.1 + 0.2 * np.random.random(np.sum(ventricle_mask))
        
        # Add cortical regions
        cortex_mask = ((x**2/0.75**2 + y**2/0.95**2 + z**2/0.65**2) < 1) & \
                     ((x**2/0.6**2 + y**2/0.8**2 + z**2/0.5**2) > 1)
        brain[cortex_mask] = 0.8 + 0.2 * np.random.random(np.sum(cortex_mask))
        
        # Apply Gaussian smoothing for realistic appearance
        brain = gaussian_filter(brain, sigma=1.5)
        
        return brain
    
    def add_ad_characteristics(self, brain):
        """Add Alzheimer's disease characteristics to the brain"""
        # Simulate brain atrophy (enlarged ventricles, reduced cortical thickness)
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, self.image_shape[0]),
            np.linspace(-1, 1, self.image_shape[1]),
            np.linspace(-1, 1, self.image_shape[2]),
            indexing='ij'
        )
        
        # Enlarged ventricles
        enlarged_ventricles = (x**2/0.25**2 + y**2/0.35**2 + z**2/0.25**2) < 1
        brain[enlarged_ventricles] *= 0.5
        
        # Reduced cortical thickness (atrophy)
        atrophy_regions = ((x**2/0.7**2 + y**2/0.9**2 + z**2/0.6**2) < 1) & \
                         ((x**2/0.55**2 + y**2/0.75**2 + z**2/0.45**2) > 1)
        brain[atrophy_regions] *= 0.7 + 0.2 * np.random.random(np.sum(atrophy_regions))
        
        # Add some random noise to simulate disease heterogeneity
        noise_mask = np.random.random(self.image_shape) < 0.1
        brain[noise_mask] *= 0.8
        
        return brain
    
    def add_normal_characteristics(self, brain):
        """Add normal brain characteristics"""
        # Add some healthy brain structure variations
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, self.image_shape[0]),
            np.linspace(-1, 1, self.image_shape[1]),
            np.linspace(-1, 1, self.image_shape[2]),
            indexing='ij'
        )
        
        # Maintain normal cortical thickness
        healthy_cortex = ((x**2/0.75**2 + y**2/0.95**2 + z**2/0.65**2) < 1) & \
                        ((x**2/0.58**2 + y**2/0.78**2 + z**2/0.48**2) > 1)
        brain[healthy_cortex] *= 1.1
        
        # Add subtle variations for realism
        variation_mask = np.random.random(self.image_shape) < 0.05
        brain[variation_mask] *= 0.95 + 0.1 * np.random.random(np.sum(variation_mask))
        
        return brain
    
    def generate_sample(self, label, subject_id):
        """Generate a single brain sample"""
        brain = self.create_brain_base()
        
        if label == "AD":
            brain = self.add_ad_characteristics(brain)
        else:  # Normal
            brain = self.add_normal_characteristics(brain)
        
        # Add noise
        brain += 0.02 * np.random.random(self.image_shape)
        
        # Normalize to typical MRI intensity range
        brain = np.clip(brain, 0, 1) * 1000
        
        # Create NIfTI image
        affine = np.eye(4)  # Identity affine matrix
        nii_img = nib.Nifti1Image(brain, affine)
        
        # Save the image
        filename = f"{subject_id}.nii"
        filepath = os.path.join(self.output_dir, filename)
        nib.save(nii_img, filepath)
        
        return filename
    
    def generate_dataset(self, n_normal=50, n_ad=50):
        """Generate a complete dataset with train/test splits"""
        # Generate data
        all_samples = []
        
        print("Generating Normal brain samples...")
        for i in range(n_normal):
            subject_id = f"NORMAL_{i:03d}"
            filename = self.generate_sample("Normal", subject_id)
            all_samples.append((filename, "Normal"))
        
        print("Generating AD brain samples...")
        for i in range(n_ad):
            subject_id = f"AD_{i:03d}"
            filename = self.generate_sample("AD", subject_id)
            all_samples.append((filename, "AD"))
        
        # Shuffle and split
        random.shuffle(all_samples)
        
        # 70% train, 15% validation, 15% test
        n_total = len(all_samples)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_samples = all_samples[:n_train]
        val_samples = all_samples[n_train:n_train + n_val]
        test_samples = all_samples[n_train + n_val:]
        
        # Write data files
        self._write_data_file("demo_train_2classes.txt", train_samples)
        self._write_data_file("demo_validation_2classes.txt", val_samples)
        self._write_data_file("demo_test_2classes.txt", test_samples)
        
        print(f"Generated {len(train_samples)} training samples")
        print(f"Generated {len(val_samples)} validation samples")
        print(f"Generated {len(test_samples)} test samples")
        
        return train_samples, val_samples, test_samples
    
    def _write_data_file(self, filename, samples):
        """Write data file in the format expected by the original code"""
        with open(filename, 'w') as f:
            for sample_file, label in samples:
                f.write(f"{sample_file} {label}\n")
    
    def visualize_sample(self, filename, save_plot=True):
        """Visualize a brain sample"""
        filepath = os.path.join(self.output_dir, filename)
        img = nib.load(filepath)
        data = img.get_fdata()
        
        # Create visualization with axial, coronal, and sagittal views
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Axial view (z=78, as used in original code)
        axes[0].imshow(data[:, :, 78], cmap='gray')
        axes[0].set_title('Axial View (z=78)')
        axes[0].axis('off')
        
        # Coronal view (y=79)
        axes[1].imshow(data[:, 79, :], cmap='gray')
        axes[1].set_title('Coronal View (y=79)')
        axes[1].axis('off')
        
        # Sagittal view (x=57)
        axes[2].imshow(data[57, :, :], cmap='gray')
        axes[2].set_title('Sagittal View (x=57)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_plot:
            plot_name = filename.replace('.nii', '_visualization.png')
            plt.savefig(plot_name, dpi=150, bbox_inches='tight')
        
        plt.show()

if __name__ == "__main__":
    # Generate demo dataset
    generator = SyntheticBrainGenerator()
    train_samples, val_samples, test_samples = generator.generate_dataset(n_normal=30, n_ad=30)
    
    # Visualize a few samples
    print("\nVisualizing sample images...")
    if train_samples:
        # Show one normal and one AD sample
        normal_sample = next(s for s in train_samples if s[1] == "Normal")
        ad_sample = next(s for s in train_samples if s[1] == "AD")
        
        print(f"Visualizing Normal sample: {normal_sample[0]}")
        generator.visualize_sample(normal_sample[0])
        
        print(f"Visualizing AD sample: {ad_sample[0]}")
        generator.visualize_sample(ad_sample[0])
    
    print("\nDemo dataset generation complete!")
    print("Files created:")
    print("- demo_train_2classes.txt")
    print("- demo_validation_2classes.txt") 
    print("- demo_test_2classes.txt")
    print("- Synthetic brain .nii files in ./demo_data/")