document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('imageUpload');
    const previewImage = document.getElementById('preview-image');
    const form = document.getElementById('upload-form');
    
    // Preview image when file is selected
    imageUpload.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                previewImage.src = e.target.result;
            };
            
            reader.readAsDataURL(this.files[0]);
        }
    });
    
    // Optional: Validate form before submission
    form.addEventListener('submit', function(e) {
        if (!imageUpload.files || imageUpload.files.length === 0) {
            e.preventDefault();
            alert('Please select an image first.');
        }
    });
});