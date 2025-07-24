import { useState } from 'react';
import {
  LoginForm,
  LoginPage,
  LoginMainFooterBandItem,
  ListVariant,
  Button,
} from '@patternfly/react-core';
import { Link } from '@tanstack/react-router';

import { ExclamationCircleIcon } from '@patternfly/react-icons';
import { useLogin } from '../hooks/useAuth';

export const SimpleLoginPage: React.FunctionComponent = () => {
  const loginMutation = useLogin();
  const [errorMessage, setErrorMessage] = useState('');

  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });

  const onLoginButtonClick = async (
    event: React.MouseEvent<HTMLButtonElement, MouseEvent>
  ) => {
    event.preventDefault();

    // Simple validation
    if (!formData.email.trim()) {
      setErrorMessage('Email is required');
      return;
    }
    if (!formData.password.trim()) {
      setErrorMessage('Password is required');
      return;
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      setErrorMessage('Please enter a valid email address');
      return;
    }

    try {
      setErrorMessage('');
      await loginMutation.mutateAsync({
        email: formData.email,
        password: formData.password,
      });
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : 'Login failed. Please try again.'
      );
    }
  };

  const signUpForAccountMessage = (
    <LoginMainFooterBandItem>
      Need an account? <Link to='/signup'>Sign up.</Link>
    </LoginMainFooterBandItem>
  );

  const forgotCredentials = (
    <LoginMainFooterBandItem>
      <div style={{ color: '#6a6e73', fontSize: '14px' }}>
        💡 Try: demo1@example.com / demo123
      </div>
      {process.env.NODE_ENV === 'development' && (
        <div style={{ marginTop: '0.5rem' }}>
          <Button
            variant='link'
            size='sm'
            onClick={() => {
              setFormData({ email: 'demo1@example.com', password: 'demo123' });
            }}
          >
            🚀 Quick Test Login
          </Button>
        </div>
      )}
    </LoginMainFooterBandItem>
  );

  const loginForm = (
    <LoginForm
      showHelperText={!!errorMessage}
      helperText={errorMessage || 'Please enter your credentials'}
      helperTextIcon={errorMessage ? <ExclamationCircleIcon /> : undefined}
      usernameLabel='Email'
      usernameValue={formData.email}
      onChangeUsername={(_event, value) => {
        setFormData(prev => ({ ...prev, email: value }));
        setErrorMessage(''); // Clear errors when typing
      }}
      isValidUsername={true}
      passwordLabel='Password'
      passwordValue={formData.password}
      onChangePassword={(_event, value) => {
        setFormData(prev => ({ ...prev, password: value }));
        setErrorMessage(''); // Clear errors when typing
      }}
      isValidPassword={true}
      onLoginButtonClick={onLoginButtonClick}
      loginButtonLabel={loginMutation.isPending ? 'Logging in...' : 'Log in'}
      isLoginButtonDisabled={loginMutation.isPending}
      rememberMeLabel='Keep me logged in'
      isRememberMeChecked={false}
      onChangeRememberMe={() => {}} // Could add remember me functionality later
    />
  );

  return (
    <LoginPage
      footerListVariants={ListVariant.inline}
      loginTitle='Log in to your account'
      loginSubtitle='Enter your credentials to access your personalized recommendations'
      signUpForAccountMessage={signUpForAccountMessage}
      forgotCredentials={forgotCredentials}
    >
      {loginForm}
    </LoginPage>
  );
};
