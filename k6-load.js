/**
 * K6 Load Testing Script for Whisper Inference Service
 * 
 * This script tests the performance of the transcription endpoint
 * under various load conditions.
 * 
 * Usage:
 *   k6 run k6-load.js
 *   k6 run --vus 10 --duration 30s k6-load.js
 *   k6 run --vus 50 --duration 2m k6-load.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const transcriptionSuccessRate = new Rate('transcription_success_rate');
const transcriptionDuration = new Trend('transcription_duration');
const transcriptionErrors = new Counter('transcription_errors');

// Test configuration
export const options = {
  stages: [
    { duration: '30s', target: 5 },   // Ramp up to 5 users
    { duration: '1m', target: 10 },   // Stay at 10 users
    { duration: '30s', target: 20 },  // Ramp up to 20 users
    { duration: '1m', target: 20 },   // Stay at 20 users
    { duration: '30s', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'], // 95% of requests must complete below 5s
    http_req_failed: ['rate<0.1'],     // Error rate must be below 10%
    transcription_success_rate: ['rate>0.9'], // Success rate must be above 90%
  },
};

// Base URL for the service
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Sample audio file data (minimal WAV file)
const SAMPLE_AUDIO = generateSampleAudio();

function generateSampleAudio() {
  // Generate a minimal WAV file header with some dummy audio data
  const sampleRate = 16000;
  const duration = 2; // 2 seconds
  const numSamples = sampleRate * duration;
  
  // WAV header (44 bytes)
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  
  // RIFF header
  view.setUint32(0, 0x46464952, false); // "RIFF"
  view.setUint32(4, 36 + numSamples * 2, true); // File size - 8
  view.setUint32(8, 0x45564157, false); // "WAVE"
  
  // fmt chunk
  view.setUint32(12, 0x20746d66, false); // "fmt "
  view.setUint32(16, 16, true); // Chunk size
  view.setUint16(20, 1, true); // Audio format (PCM)
  view.setUint16(22, 1, true); // Number of channels
  view.setUint32(24, sampleRate, true); // Sample rate
  view.setUint32(28, sampleRate * 2, true); // Byte rate
  view.setUint16(32, 2, true); // Block align
  view.setUint16(34, 16, true); // Bits per sample
  
  // data chunk
  view.setUint32(36, 0x61746164, false); // "data"
  view.setUint32(40, numSamples * 2, true); // Data size
  
  // Create audio data (simple sine wave)
  const audioData = new ArrayBuffer(numSamples * 2);
  const audioView = new DataView(audioData);
  for (let i = 0; i < numSamples; i++) {
    const sample = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 32767; // 440Hz tone
    audioView.setInt16(i * 2, sample, true);
  }
  
  // Combine header and audio data
  const fullAudio = new Uint8Array(44 + numSamples * 2);
  fullAudio.set(new Uint8Array(header), 0);
  fullAudio.set(new Uint8Array(audioData), 44);
  
  return fullAudio.buffer;
}

export function setup() {
  // Check if service is healthy before starting load test
  const healthResponse = http.get(`${BASE_URL}/healthz`);
  check(healthResponse, {
    'health check passed': (r) => r.status === 200,
  });
  
  // Check if service is ready
  const readyResponse = http.get(`${BASE_URL}/readyz`);
  check(readyResponse, {
    'readiness check passed': (r) => r.status === 200,
  });
  
  console.log('Service is healthy and ready for load testing');
}

export default function() {
  // Test health endpoint
  testHealthEndpoint();
  
  // Test readiness endpoint
  testReadinessEndpoint();
  
  // Test metrics endpoint
  testMetricsEndpoint();
  
  // Test transcription endpoint
  testTranscriptionEndpoint();
  
  // Wait between requests
  sleep(1);
}

function testHealthEndpoint() {
  const response = http.get(`${BASE_URL}/healthz`);
  
  check(response, {
    'health status is 200': (r) => r.status === 200,
    'health response time < 100ms': (r) => r.timings.duration < 100,
    'health response has status field': (r) => JSON.parse(r.body).status === 'healthy',
  });
}

function testReadinessEndpoint() {
  const response = http.get(`${BASE_URL}/readyz`);
  
  check(response, {
    'readiness status is 200': (r) => r.status === 200,
    'readiness response time < 100ms': (r) => r.timings.duration < 100,
    'readiness response has status field': (r) => JSON.parse(r.body).status === 'ready',
  });
}

function testMetricsEndpoint() {
  const response = http.get(`${BASE_URL}/metrics`);
  
  check(response, {
    'metrics status is 200': (r) => r.status === 200,
    'metrics response time < 100ms': (r) => r.timings.duration < 100,
    'metrics content type is text/plain': (r) => r.headers['Content-Type'].includes('text/plain'),
    'metrics contains transcription_requests_total': (r) => r.body.includes('transcription_requests_total'),
  });
}

function testTranscriptionEndpoint() {
  const startTime = Date.now();
  
  // Prepare form data
  const formData = {
    file: http.file(SAMPLE_AUDIO, 'test.wav', 'audio/wav'),
  };
  
  const params = {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  };
  
  const response = http.post(`${BASE_URL}/transcribe`, formData, params);
  const duration = Date.now() - startTime;
  
  const success = check(response, {
    'transcription status is 200': (r) => r.status === 200,
    'transcription response time < 30s': (r) => r.timings.duration < 30000,
    'transcription has filename': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.filename === 'test.wav';
      } catch (e) {
        return false;
      }
    },
    'transcription has language': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.language !== undefined;
      } catch (e) {
        return false;
      }
    },
    'transcription has full_text': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.transcription && body.transcription.full_text !== undefined;
      } catch (e) {
        return false;
      }
    },
  });
  
  // Record custom metrics
  transcriptionSuccessRate.add(success);
  transcriptionDuration.add(duration);
  
  if (!success) {
    transcriptionErrors.add(1);
    console.error(`Transcription failed: ${response.status} - ${response.body}`);
  }
}

export function teardown(data) {
  console.log('Load test completed');
  console.log('Check the metrics endpoint for detailed performance data');
}
