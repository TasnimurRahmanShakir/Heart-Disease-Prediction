# MedScan React Frontend

A modern React application for the Heart Disease Prediction system with beautiful UI and advanced features.

## Features

- **Modern React 18** with TypeScript
- **Drag & Drop** file uploads using react-dropzone
- **Real-time Notifications** with react-hot-toast
- **Responsive Design** that works on all devices
- **Beautiful UI** with glassmorphism effects
- **Type Safety** with TypeScript
- **Component-based Architecture** for maintainability

## Tech Stack

- **React 18** - UI Framework
- **TypeScript** - Type Safety
- **Axios** - HTTP Client
- **React Dropzone** - File Upload
- **React Hot Toast** - Notifications
- **CSS3** - Styling with modern features

## Getting Started

### Prerequisites

- Node.js 16+ and npm
- Backend server running on `http://localhost:8000`

### Installation

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend-react
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```

4. **Open your browser:**
   Navigate to `http://localhost:3000`

### Quick Start (Windows)

Double-click `start_frontend_react.bat` from the project root.

## Project Structure

```
frontend-react/
├── public/
│   └── index.html          # Main HTML file
├── src/
│   ├── components/         # React components
│   │   ├── Header.tsx      # Header component
│   │   ├── Header.css      # Header styles
│   │   ├── PredictionTabs.tsx
│   │   ├── PredictionTabs.css
│   │   ├── SingleImagePrediction.tsx
│   │   ├── SingleImagePrediction.css
│   │   ├── BatchPrediction.tsx
│   │   └── BatchPrediction.css
│   ├── App.tsx             # Main App component
│   ├── App.css             # Global styles
│   └── index.tsx           # Entry point
├── package.json            # Dependencies
└── tsconfig.json           # TypeScript config
```

## Components

### Header
- Animated heartbeat logo
- Application title and tagline
- Responsive design

### PredictionTabs
- Tab navigation between single and batch modes
- Smooth transitions and hover effects

### SingleImagePrediction
- Drag & drop image upload
- Image preview with remove functionality
- Real-time prediction results
- Confidence scores with color coding

### BatchPrediction
- ZIP file upload for batch processing
- Statistics display (total, positive, negative cases)
- Download links for confusion matrix and ROC curve

## API Integration

The frontend communicates with the backend API:

- **Single Prediction:** `POST /api/predict/single`
- **Batch Prediction:** `POST /api/predict/folder`
- **Download Confusion Matrix:** `GET /api/download/confusion_matrix`
- **Download ROC Curve:** `GET /api/download/roc_curve`

## Styling

- **Modern CSS** with flexbox and grid
- **Glassmorphism effects** with backdrop-filter
- **Smooth animations** and transitions
- **Responsive design** for mobile and desktop
- **Color-coded results** (red for positive, green for negative)

## Development

### Available Scripts

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App

### Adding New Features

1. **Create new components** in `src/components/`
2. **Add TypeScript interfaces** for type safety
3. **Style with CSS modules** or component-specific CSS
4. **Update API calls** in the relevant component

### Customization

- **Colors:** Update CSS variables in component files
- **Styling:** Modify CSS files for visual changes
- **Functionality:** Add new features in TypeScript components
- **API:** Update axios calls for backend changes

## Production Build

1. **Build the application:**
   ```bash
   npm run build
   ```

2. **Serve the build folder:**
   ```bash
   npx serve -s build
   ```

## Troubleshooting

### Common Issues

1. **Backend not running:**
   - Ensure backend is running on `http://localhost:8000`
   - Check console for CORS errors

2. **File upload not working:**
   - Check file type restrictions
   - Ensure file size is within limits

3. **Styling issues:**
   - Clear browser cache
   - Check CSS file imports

### Development Tips

- Use React Developer Tools for debugging
- Check browser console for errors
- Test on different screen sizes
- Verify API responses in Network tab

## Contributing

1. Follow TypeScript best practices
2. Use meaningful component and variable names
3. Add proper error handling
4. Test on multiple browsers
5. Ensure responsive design works

## License

This project is part of the MedScan Heart Disease Prediction system.
