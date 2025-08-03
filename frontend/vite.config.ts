import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { tanstackRouter } from '@tanstack/router-plugin/vite';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    // Please make sure that '@tanstack/router-plugin' is passed before '@vitejs/plugin-react'
    tanstackRouter({ target: 'react', autoCodeSplitting: true }),
    react(),
    // ...,
  ],
  resolve: {
    alias: {
      '@': resolve(process.cwd(), './src'), // This line maps '@/' to your 'src' directory
    },
  },
  server: {
    proxy: {
      '/auth': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/products': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/recommendations': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/users': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/interactions': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/cart': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/orders': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/checkout': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/wishlist': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/feedback': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: { chunkSizeWarningLimit: 2000 },
});
