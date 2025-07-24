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
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, ''),
      },
    },
  },
  build: { chunkSizeWarningLimit: 2000 },
});
