"""
Homomorphic Encryption Module for Bharat-FM
Implements privacy-preserving computation on encrypted data
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle
import hashlib
import os
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EncryptionConfig:
    """Configuration for homomorphic encryption"""
    scheme: str = "ckks"  # CKKS for approximate arithmetic
    poly_modulus_degree: int = 8192
    coeff_modulus_bits: List[int] = None
    scale: float = 2**40
    security_level: int = 128
    
    def __post_init__(self):
        if self.coeff_modulus_bits is None:
            self.coeff_modulus_bits = [60, 40, 40, 60]

class HomomorphicEncryptor:
    """
    Implements homomorphic encryption for secure computation
    Supports CKKS scheme for approximate arithmetic on real numbers
    """
    
    def __init__(self, config: EncryptionConfig = None):
        self.config = config or EncryptionConfig()
        self.context = None
        self.key_generator = None
        self.public_key = None
        self.secret_key = None
        self.relin_keys = None
        self.galois_keys = None
        self._initialize_scheme()
    
    def _initialize_scheme(self):
        """Initialize the encryption scheme"""
        try:
            # Try to use SEAL (Microsoft Simple Encrypted Arithmetic Library)
            self._initialize_seal()
        except ImportError:
            logger.warning("SEAL not available, falling back to simulation mode")
            self._initialize_simulation()
    
    def _initialize_seal(self):
        """Initialize using Microsoft SEAL library"""
        try:
            import seal
            from seal import EncryptionParameters, SchemeType, CoeffModulus, \
                            SEALContext, KeyGenerator, Encryptor, Decryptor, \
                            Evaluator, CKKSEncoder, FractionalEncoder
            
            # Set up encryption parameters
            parms = EncryptionParameters(SchemeType.CKKS)
            parms.set_poly_modulus_degree(self.config.poly_modulus_degree)
            parms.set_coeff_modulus(CoeffModulus.Create(
                self.config.poly_modulus_degree, 
                self.config.coeff_modulus_bits
            ))
            
            # Create context
            self.context = SEALContext(parms)
            
            # Generate keys
            self.key_generator = KeyGenerator(self.context)
            self.public_key = self.key_generator.public_key()
            self.secret_key = self.key_generator.secret_key()
            
            # Create additional keys for computations
            self.relin_keys = self.key_generator.relin_keys()
            self.galois_keys = self.key_generator.galois_keys()
            
            # Create encoder and evaluator
            self.encoder = CKKSEncoder(self.context)
            self.evaluator = Evaluator(self.context)
            self.encryptor = Encryptor(self.context, self.public_key)
            self.decryptor = Decryptor(self.context, self.secret_key)
            
            self.scheme = "seal_ckks"
            logger.info("SEAL CKKS scheme initialized successfully")
            
        except ImportError:
            raise ImportError("Microsoft SEAL not available")
    
    def _initialize_simulation(self):
        """Initialize simulation mode for development/testing"""
        self.scheme = "simulation"
        self.simulation_key = os.urandom(32)
        logger.info("Simulation mode initialized")
    
    def encrypt_vector(self, vector: np.ndarray) -> Any:
        """
        Encrypt a vector of numbers
        Returns encrypted ciphertext
        """
        if self.scheme == "seal_ckks":
            return self._encrypt_vector_seal(vector)
        else:
            return self._encrypt_vector_simulation(vector)
    
    def _encrypt_vector_seal(self, vector: np.ndarray) -> Any:
        """Encrypt vector using SEAL"""
        # Encode the vector
        plain_vector = self.encoder.encode(vector, self.config.scale)
        
        # Encrypt the encoded vector
        encrypted_vector = self.encryptor.encrypt(plain_vector)
        
        return encrypted_vector
    
    def _encrypt_vector_simulation(self, vector: np.ndarray) -> Dict:
        """Simulate encryption for development"""
        # In simulation mode, we just store the vector with a checksum
        checksum = hashlib.sha256(pickle.dumps(vector)).hexdigest()
        return {
            'data': vector.tolist(),
            'checksum': checksum,
            'scheme': 'simulation'
        }
    
    def decrypt_vector(self, ciphertext: Any) -> np.ndarray:
        """
        Decrypt a ciphertext back to vector
        """
        if self.scheme == "seal_ckks":
            return self._decrypt_vector_seal(ciphertext)
        else:
            return self._decrypt_vector_simulation(ciphertext)
    
    def _decrypt_vector_seal(self, ciphertext: Any) -> np.ndarray:
        """Decrypt vector using SEAL"""
        # Decrypt the ciphertext
        plain_vector = self.decryptor.decrypt(ciphertext)
        
        # Decode the result
        result = self.encoder.decode(plain_vector)
        
        return np.array(result)
    
    def _decrypt_vector_simulation(self, ciphertext: Dict) -> np.ndarray:
        """Simulate decryption for development"""
        vector = np.array(ciphertext['data'])
        checksum = hashlib.sha256(pickle.dumps(vector)).hexdigest()
        
        if checksum != ciphertext['checksum']:
            raise ValueError("Checksum verification failed")
        
        return vector
    
    def add_ciphertexts(self, ciphertext1: Any, ciphertext2: Any) -> Any:
        """
        Add two encrypted vectors
        """
        if self.scheme == "seal_ckks":
            result = self.evaluator.add(ciphertext1, ciphertext2)
            return result
        else:
            # Simulation mode
            vec1 = np.array(ciphertext1['data'])
            vec2 = np.array(ciphertext2['data'])
            result_vec = vec1 + vec2
            
            checksum = hashlib.sha256(pickle.dumps(result_vec)).hexdigest()
            return {
                'data': result_vec.tolist(),
                'checksum': checksum,
                'scheme': 'simulation'
            }
    
    def multiply_ciphertexts(self, ciphertext1: Any, ciphertext2: Any) -> Any:
        """
        Multiply two encrypted vectors
        """
        if self.scheme == "seal_ckks":
            result = self.evaluator.multiply(ciphertext1, ciphertext2)
            self.evaluator.relinearize_inplace(result, self.relin_keys)
            return result
        else:
            # Simulation mode
            vec1 = np.array(ciphertext1['data'])
            vec2 = np.array(ciphertext2['data'])
            result_vec = vec1 * vec2
            
            checksum = hashlib.sha256(pickle.dumps(result_vec)).hexdigest()
            return {
                'data': result_vec.tolist(),
                'checksum': checksum,
                'scheme': 'simulation'
            }
    
    def scalar_multiply(self, ciphertext: Any, scalar: float) -> Any:
        """
        Multiply encrypted vector by scalar
        """
        if self.scheme == "seal_ckks":
            # Encode scalar
            plain_scalar = self.encoder.encode(scalar, self.config.scale)
            
            # Multiply
            result = self.evaluator.multiply_plain(ciphertext, plain_scalar)
            return result
        else:
            # Simulation mode
            vec = np.array(ciphertext['data'])
            result_vec = vec * scalar
            
            checksum = hashlib.sha256(pickle.dumps(result_vec)).hexdigest()
            return {
                'data': result_vec.tolist(),
                'checksum': checksum,
                'scheme': 'simulation'
            }
    
    def rotate_vector(self, ciphertext: Any, steps: int) -> Any:
        """
        Rotate encrypted vector (useful for convolutions)
        """
        if self.scheme == "seal_ckks":
            result = self.evaluator.rotate_vector(ciphertext, steps, self.galois_keys)
            return result
        else:
            # Simulation mode
            vec = np.array(ciphertext['data'])
            result_vec = np.roll(vec, steps)
            
            checksum = hashlib.sha256(pickle.dumps(result_vec)).hexdigest()
            return {
                'data': result_vec.tolist(),
                'checksum': checksum,
                'scheme': 'simulation'
            }
    
    def save_keys(self, filepath: str):
        """Save encryption keys to file"""
        keys_data = {
            'scheme': self.scheme,
            'config': self.config.__dict__,
            'simulation_key': self.simulation_key if self.scheme == 'simulation' else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(keys_data, f)
        
        logger.info(f"Keys saved to {filepath}")
    
    def load_keys(self, filepath: str):
        """Load encryption keys from file"""
        with open(filepath, 'rb') as f:
            keys_data = pickle.load(f)
        
        self.scheme = keys_data['scheme']
        self.config = EncryptionConfig(**keys_data['config'])
        
        if self.scheme == 'simulation':
            self.simulation_key = keys_data['simulation_key']
        
        logger.info(f"Keys loaded from {filepath}")

class SecureMLModel:
    """
    Secure machine learning model using homomorphic encryption
    Enables privacy-preserving inference
    """
    
    def __init__(self, encryptor: HomomorphicEncryptor):
        self.encryptor = encryptor
        self.weights = None
        self.bias = None
        self.model_type = None
    
    def encrypt_model(self, weights: np.ndarray, bias: np.ndarray = None):
        """
        Encrypt model parameters for secure inference
        """
        self.weights = self.encryptor.encrypt_vector(weights)
        if bias is not None:
            self.bias = self.encryptor.encrypt_vector(bias)
        self.model_type = "linear"
    
    def secure_inference(self, encrypted_input: Any) -> Any:
        """
        Perform secure inference on encrypted input
        """
        if self.model_type == "linear":
            return self._secure_linear_inference(encrypted_input)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _secure_linear_inference(self, encrypted_input: Any) -> Any:
        """
        Secure linear inference: y = Wx + b
        """
        # Compute matrix-vector multiplication
        result = self.encryptor.multiply_ciphertexts(encrypted_input, self.weights)
        
        # Add bias if available
        if self.bias is not None:
            result = self.encryptor.add_ciphertexts(result, self.bias)
        
        return result
    
    def secure_activation(self, encrypted_output: Any, activation: str = "relu") -> Any:
        """
        Apply activation function to encrypted output
        Note: Some activations may require approximation
        """
        if activation == "relu":
            # ReLU approximation using polynomial
            return self._approximate_relu(encrypted_output)
        elif activation == "sigmoid":
            # Sigmoid approximation
            return self._approximate_sigmoid(encrypted_output)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _approximate_relu(self, x: Any) -> Any:
        """
        Approximate ReLU using polynomial: relu(x) ≈ 0.5 * (x + sqrt(x²))
        """
        # Compute x²
        x_squared = self.encryptor.multiply_ciphertexts(x, x)
        
        # For simulation, we can compute sqrt directly
        # In real implementation, this would require more sophisticated approximation
        if self.encryptor.scheme == "simulation":
            x_vec = np.array(x['data'])
            relu_approx = 0.5 * (x_vec + np.sqrt(np.maximum(x_vec**2, 0)))
            
            checksum = hashlib.sha256(pickle.dumps(relu_approx)).hexdigest()
            return {
                'data': relu_approx.tolist(),
                'checksum': checksum,
                'scheme': 'simulation'
            }
        else:
            # In real implementation, this would use polynomial approximation
            raise NotImplementedError("ReLU approximation not implemented for SEAL")
    
    def _approximate_sigmoid(self, x: Any) -> Any:
        """
        Approximate sigmoid using polynomial
        """
        if self.encryptor.scheme == "simulation":
            x_vec = np.array(x['data'])
            # Simple polynomial approximation of sigmoid
            sigmoid_approx = 0.5 + 0.25 * x_vec - 0.01 * x_vec**3
            
            checksum = hashlib.sha256(pickle.dumps(sigmoid_approx)).hexdigest()
            return {
                'data': sigmoid_approx.tolist(),
                'checksum': checksum,
                'scheme': 'simulation'
            }
        else:
            raise NotImplementedError("Sigmoid approximation not implemented for SEAL")

# Factory function for creating encryptors
def create_encryptor(scheme: str = "ckks", **kwargs) -> HomomorphicEncryptor:
    """
    Create a homomorphic encryptor with specified configuration
    """
    config = EncryptionConfig(scheme=scheme, **kwargs)
    return HomomorphicEncryptor(config)

# Example usage and testing
def test_homomorphic_encryption():
    """Test the homomorphic encryption functionality"""
    print("Testing Homomorphic Encryption...")
    
    # Create encryptor
    encryptor = create_encryptor()
    
    # Test data
    vector1 = np.array([1.0, 2.0, 3.0, 4.0])
    vector2 = np.array([0.5, 1.5, 2.5, 3.5])
    
    # Encrypt vectors
    print("Encrypting vectors...")
    encrypted1 = encryptor.encrypt_vector(vector1)
    encrypted2 = encryptor.encrypt_vector(vector2)
    
    # Perform operations
    print("Performing homomorphic operations...")
    
    # Addition
    encrypted_sum = encryptor.add_ciphertexts(encrypted1, encrypted2)
    decrypted_sum = encryptor.decrypt_vector(encrypted_sum)
    expected_sum = vector1 + vector2
    
    print(f"Addition test: {np.allclose(decrypted_sum, expected_sum)}")
    
    # Multiplication
    encrypted_product = encryptor.multiply_ciphertexts(encrypted1, encrypted2)
    decrypted_product = encryptor.decrypt_vector(encrypted_product)
    expected_product = vector1 * vector2
    
    print(f"Multiplication test: {np.allclose(decrypted_product, expected_product)}")
    
    # Scalar multiplication
    scalar = 2.0
    encrypted_scaled = encryptor.scalar_multiply(encrypted1, scalar)
    decrypted_scaled = encryptor.decrypt_vector(encrypted_scaled)
    expected_scaled = vector1 * scalar
    
    print(f"Scalar multiplication test: {np.allclose(decrypted_scaled, expected_scaled)}")
    
    # Test secure ML model
    print("Testing secure ML model...")
    secure_model = SecureMLModel(encryptor)
    
    # Simple linear model
    weights = np.array([0.5, -0.3, 0.8, -0.1])
    bias = np.array([0.1])
    
    secure_model.encrypt_model(weights, bias)
    
    # Secure inference
    encrypted_input = encryptor.encrypt_vector(vector1)
    encrypted_output = secure_model.secure_inference(encrypted_input)
    decrypted_output = encryptor.decrypt_vector(encrypted_output)
    
    # Compare with plaintext computation
    expected_output = np.dot(vector1, weights) + bias
    print(f"Secure inference test: {np.allclose(decrypted_output, expected_output)}")
    
    print("Homomorphic encryption tests completed!")

if __name__ == "__main__":
    test_homomorphic_encryption()