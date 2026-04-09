import { SignUp } from '@clerk/react';

export default function SignUpPage() {
  return (
    <div
      className="min-h-screen flex items-center justify-center px-4 py-24"
      style={{ background: 'linear-gradient(175deg, #e8e4f0 0%, #edf0f4 40%, #F8F4EA 100%)' }}
    >
      <SignUp routing="path" path="/sign-up" signInUrl="/sign-in" />
    </div>
  );
}
