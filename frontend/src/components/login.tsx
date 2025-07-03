import { useState } from "react";
import {
  LoginForm,
  LoginMainFooterBandItem,
  LoginPage,
  ListVariant,
} from "@patternfly/react-core";
import ExclamationCircleIcon from "@patternfly/react-icons/dist/esm/icons/exclamation-circle-icon";
import { Link } from "@tanstack/react-router";

export const SimpleLoginPage: React.FunctionComponent = () => {
  const [showHelperText, setShowHelperText] = useState(false);
  const [username, setUsername] = useState("");
  const [isValidUsername, setIsValidUsername] = useState(true);
  const [password, setPassword] = useState("");
  const [isValidPassword, setIsValidPassword] = useState(true);

  const handleUsernameChange = (
    _event: React.FormEvent<HTMLInputElement>,
    value: string
  ) => {
    setUsername(value);
  };

  const handlePasswordChange = (
    _event: React.FormEvent<HTMLInputElement>,
    value: string
  ) => {
    setPassword(value);
  };

  const onLoginButtonClick = (
    event: React.MouseEvent<HTMLButtonElement, MouseEvent>
  ) => {
    event.preventDefault();
    setIsValidUsername(!!username);
    setIsValidPassword(!!password);
    setShowHelperText(!username || !password);
  };

  const signUpForAccountMessage = (
    <LoginMainFooterBandItem>
      Need an account? <Link to="/signup">Sign up.</Link>
    </LoginMainFooterBandItem>
  );

  const forgotCredentials = (
    <LoginMainFooterBandItem>
      {/* <a href="">Forgot username or password?</a> */}
    </LoginMainFooterBandItem>
  );

  const loginForm = (
    <LoginForm
      showHelperText={showHelperText}
      helperText="Invalid login credentials."
      helperTextIcon={<ExclamationCircleIcon />}
      usernameLabel="Email/UserID"
      usernameValue={username}
      onChangeUsername={handleUsernameChange}
      isValidUsername={isValidUsername}
      passwordLabel="Password"
      passwordValue={password}
      onChangePassword={handlePasswordChange}
      isValidPassword={isValidPassword}
      onLoginButtonClick={onLoginButtonClick}
      loginButtonLabel="Log in"
    />
  );

  return (
    <LoginPage
      footerListVariants={ListVariant.inline}
      loginTitle="Log in to your account"
      signUpForAccountMessage={signUpForAccountMessage}
      forgotCredentials={forgotCredentials}
    >
      {loginForm}
    </LoginPage>
  );
};
